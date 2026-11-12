"""Offline source-anchored preference correction for the original MDR bank.

Uses only TRAIN history/next-item pairs in the existing frozen MDR encoder
space. Retrieval keys and source variances stay fixed. No neural loss, extra
training stage, CF graph, or denoiser conditioning is introduced.
"""
import torch
import torch.nn.functional as F


def supported_target_labels(target_subset, target_train, item_count):
    """Exclude untrained target-label embeddings from the offline fit only.

    Before target fine-tuning, target embeddings receive source-stage gradients
    through TRAIN histories. A label absent from all such histories must not
    become arbitrary next-preference supervision. Training/evaluation rows are
    never removed; this mask applies only to the offline correction fit.
    """
    seen = torch.zeros(item_count, dtype=torch.bool)
    sequence = torch.as_tensor(list(target_train['seq'].values), dtype=torch.long)
    seen[sequence.reshape(-1)] = True
    seen[0] = False
    labels = torch.as_tensor(target_subset['next'].to_numpy(), dtype=torch.long)
    if bool(((labels <= 0) | (labels >= item_count)).any()):
        raise ValueError('Invalid target TRAIN next-item ID')
    return seen[labels]


@torch.no_grad()
def next_item_preferences(model, train, source, batch_size, width, device):
    """Encode TRAIN labels as right-aligned, one-item histories in MDR space."""
    table = model.souce_embeddings if source else model.target_embeddings
    model.eval()
    positions = torch.arange(width, device=device)
    chunks = []
    for start in range(0, len(train), batch_size):
        labels = torch.as_tensor(train['next'].iloc[start:start + batch_size].to_numpy(),
                                 device=device, dtype=torch.long)
        if bool(((labels <= 0) | (labels >= table.num_embeddings)).any()):
            raise ValueError('Invalid TRAIN next-item ID in preference correction')
        sequence = torch.zeros((len(labels), width), device=device, dtype=torch.long)
        sequence[:, -1] = labels
        embeddings = table(sequence) + model.position_embeddings(positions)[None, :, :]
        chunks.append(model.prototype_mdr.encode(embeddings, (sequence > 0).float()))
    return torch.cat(chunks)


def retrieval_weights(query, memory):
    """Match the original sampler's Top-K/publicness retrieval without RNG."""
    public = memory.publicness().detach()
    valid = (memory.source_seen > 0) & (public >= memory.min_public)
    count = int(valid.sum())
    if not count:
        return query.new_zeros((len(query), memory.num_prototypes))
    logits = query @ F.normalize(memory.prototypes.detach(), dim=-1).T
    logits = logits / memory.retrieval_temperature
    logits = logits + memory.public_weight * public.clamp_min(1e-8).log()
    logits = logits.masked_fill(~valid[None, :], -1e9)
    scores, indices = logits.topk(min(memory.topk, count), dim=-1)
    return torch.zeros_like(logits).scatter(1, indices, scores.softmax(-1))


def corrected_means(query, indices, means, delta):
    """Apply z(h)(W_target - W_source) to the selected original source means."""
    z = torch.cat((query, torch.ones_like(query[:, :1])), dim=-1)
    shifts = [torch.bmm(z[:, None, :], delta[indices[:, j]]).squeeze(1)
              for j in range(indices.size(1))]
    return means + torch.stack(shifts, dim=1)


def _design(query):
    query = query.double()
    return torch.cat((query, torch.ones_like(query[:, :1])), dim=-1)


def _statistics(z, weights, future):
    # Top-K routes are sparse. Per-key matrix products avoid an N x K x D
    # intermediate, which would consume several GB on the real datasets.
    future = future.double()
    gram = z.new_zeros((weights.size(1), z.size(1), z.size(1)))
    cross = z.new_zeros((weights.size(1), z.size(1), future.size(1)))
    for key in range(weights.size(1)):
        active = weights[:, key] > 0
        if not bool(active.any()):
            continue
        w = weights[active, key].double()
        local = z[active]
        mass = w.sum().clamp_min(1e-12)
        gram[key] = local.T @ (w[:, None] * local) / mass
        cross[key] = local.T @ (w[:, None] * future[active]) / mass
    return gram, cross


def _solve(gram, cross, ridge, prior=None):
    eye = torch.eye(gram.size(-1), device=gram.device, dtype=gram.dtype)
    return torch.linalg.solve(gram + ridge * eye,
                              cross if prior is None else cross + ridge * prior)


def _metrics(z, weights, means, delta, future):
    mse, cosine = 0.0, 0.0
    for start in range(0, len(z), 512):
        w = weights[start:start + 512].double()
        prediction = w @ means.double()
        per_key = torch.einsum('ni,kij->nkj', z[start:start + 512], delta)
        prediction += torch.einsum('nk,nkj->nj', w, per_key)
        label = future[start:start + 512].double()
        mse += float((prediction - label).square().sum())
        cosine += float(F.cosine_similarity(prediction, label, dim=-1).sum())
    return dict(mse=mse / len(z), cosine=cosine / len(z))


@torch.no_grad()
def fit_preference_delta(source, source_future, target, target_future, memory,
                         supported, seed, source_ridge=0.01,
                         target_ridges=(0.001, 0.01, 0.1, 1.0, 10.0)):
    """Fit local functions and choose target shrinkage on an internal TRAIN split.

    For each original prototype k:
        W_t,k = (G_t,k + lambda I)^-1(C_t,k + lambda W_s,k).
    The existing sampler mean changes by z(h)(W_t,k - W_s,k). A no-change
    candidate participates in selection. The selected lambda is then refitted
    on the whole fixed target TRAIN subset; no outer val/test row is accepted.
    """
    if source_ridge <= 0 or not target_ridges or any(r <= 0 for r in target_ridges):
        raise ValueError('Preference ridge values must be positive')
    if source.shape != source_future.shape or target.shape != target_future.shape:
        raise ValueError('History and next-item encoder shapes must match')
    if not all(torch.isfinite(x).all() for x in (source, source_future, target, target_future)):
        raise ValueError('Preference fit contains nonfinite encoder features')
    zero = memory.source_mean.new_zeros((memory.num_prototypes, source.size(1) + 1, source.size(1)))
    report = dict(method='source_anchored_preference_ridge', source_rows=len(source),
                  target_subset_rows=len(target), source_ridge=source_ridge,
                  target_ridges=list(target_ridges), enabled=False,
                  feature_space='original MDR encoder; right-aligned one-item TRAIN next preference',
                  source_labels_used=True, target_labels_used=True, test_read=False,
                  selection='fixed 80/20 split of target TRAIN subset; sampler-mean MSE; no-change candidate included')
    if not len(source) or len(target) < 5:
        return zero, dict(report, reason='insufficient training rows')
    source_weights = retrieval_weights(source, memory)
    target_weights = retrieval_weights(target, memory)
    eligible = supported & (target_weights.sum(0) > 0)
    if not bool(eligible.any()):
        return zero, dict(report, reason='no supported retrievable prototypes')
    zs, zt = _design(source), _design(target)
    gs, cs = _statistics(zs, source_weights, source_future)
    global_model = _solve(zs.T @ zs / len(zs), zs.T @ source_future.double() / len(zs), source_ridge)
    source_models = _solve(gs, cs, source_ridge, global_model)
    generator = torch.Generator().manual_seed(int(seed) + 2711)
    order = torch.randperm(len(target), generator=generator).to(target.device)
    nfit = min(len(target) - 1, max(1, int(0.8 * len(target))))
    fit, hold = order[:nfit], order[nfit:]
    gram, cross = _statistics(zt[fit], target_weights[fit], target_future[fit])
    no_change = torch.zeros_like(source_models)
    before = _metrics(zt[hold], target_weights[hold], memory.source_mean, no_change, target_future[hold])
    best, selected = before['mse'], None
    records = [dict(ridge=None, **before)]
    for ridge in target_ridges:
        delta = _solve(gram, cross, ridge, source_models) - source_models
        delta[~eligible] = 0
        metrics = _metrics(zt[hold], target_weights[hold], memory.source_mean, delta, target_future[hold])
        records.append(dict(ridge=ridge, **metrics))
        if metrics['mse'] < best:
            best, selected = metrics['mse'], float(ridge)
    report.update(fit_rows=len(fit), holdout_rows=len(hold), eligible_keys=int(eligible.sum()),
                  selected_ridge=selected, holdout_before=before, grid=records)
    if selected is None:
        return zero, dict(report, holdout_after=before, reason='no-change candidate selected')
    gram, cross = _statistics(zt, target_weights, target_future)
    delta = _solve(gram, cross, selected, source_models) - source_models
    delta[~eligible] = 0
    report.update(enabled=True, corrected_keys=int(eligible.sum()),
                  holdout_after=next(r for r in records if r['ridge'] == selected),
                  mean_delta_norm=float(delta.norm(dim=(1, 2)).mean()),
                  final_refit='all rows in the same fixed target TRAIN subset')
    return delta.to(dtype=zero.dtype), report
