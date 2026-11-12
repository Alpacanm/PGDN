"""Optional, post-hoc target-domain score correction.

The pretrained diffusion model and target item table are frozen.  This module
only changes the query used by the final target-item dot product.
"""

import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from trainer import hrs_and_ndcgs_k, mask_padding_item


class TargetResidualScorer(nn.Module):
    """Recover target-history features weakly represented in diffusion output."""

    def __init__(self, item_vectors, sparse_multiplier=4, sparse_topk=32):
        super().__init__()
        if item_vectors.ndim != 2 or item_vectors.size(0) < 3:
            raise ValueError('Expected a target item table including padding and two items')
        hidden_size = item_vectors.size(1)
        sparse_size = hidden_size * sparse_multiplier
        if not 0 < sparse_topk <= sparse_size:
            raise ValueError('sparse_topk must be within the sparse feature size')

        # The checkpoint already stores this frozen table in the base model.
        self.register_buffer('item_vectors', item_vectors.detach().clone(),
                             persistent=False)
        self.sparse_topk = sparse_topk
        self.target_encoder = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.sparse_encoder = nn.Linear(hidden_size, sparse_size)
        self.decoder = nn.Linear(sparse_size, hidden_size, bias=False)
        nn.init.normal_(self.decoder.weight, std=0.01)
        self.mix_logit = nn.Parameter(torch.tensor(-2.2))

    def _sparse(self, representation):
        activations = F.relu(self.sparse_encoder(representation))
        values, indices = activations.topk(self.sparse_topk, dim=-1)
        sparse = torch.zeros_like(activations)
        return sparse.scatter(1, indices, values)

    def query(self, sequence, diffusion_rep, enabled=True):
        if not enabled:
            return diffusion_rep, None, None

        history = F.embedding(sequence, self.item_vectors)
        mask = (sequence > 0).unsqueeze(-1)
        mean = (history * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        last = history[:, -1]
        target_rep = self.target_encoder(torch.cat([mean, last], dim=-1))

        diffusion_features = self._sparse(diffusion_rep)
        target_features = self._sparse(target_rep)
        target_only = F.relu(target_features - diffusion_features)
        correction = self.decoder(target_only)
        mixed = diffusion_rep + torch.sigmoid(self.mix_logit) * correction
        reconstruction = (
            F.mse_loss(self.decoder(diffusion_features), diffusion_rep)
            + F.mse_loss(self.decoder(target_features), target_rep)
        )
        return mixed, target_rep, reconstruction

    def scores(self, query, candidate_ids=None):
        if candidate_ids is None:
            return query @ self.item_vectors.t()
        candidates = F.embedding(candidate_ids, self.item_vectors)
        return (query.unsqueeze(1) * candidates).sum(dim=-1)


def _cache_inference_representations(model, data, args, batch_size, name):
    """Use the label-free reverse path used at validation and test time."""
    model.eval()
    sequences = []
    representations = []
    labels = []
    allowed_rows = len(data)
    if name != 'train' and args.evaluation_protocol == 'd2tcdr_legacy':
        # Match the original evaluator's rows, independently of cache batch size.
        allowed_rows = (allowed_rows // args.batch_size) * args.batch_size
    starts = range(0, allowed_rows, batch_size)
    with torch.no_grad():
        for start in starts:
            batch = data.iloc[start:min(start + batch_size, allowed_rows)]
            sequence = torch.as_tensor(np.stack(batch['seq'].to_numpy()),
                                       dtype=torch.long, device=args.device)
            # The inference branch does not read `tag`; zero is deliberately passed.
            dummy_tag = torch.zeros((len(batch), 1), dtype=torch.long,
                                    device=args.device)
            _, diffusion_rep, _, _, _, _, _ = model(
                sequence, dummy_tag, None, False, args, 0, train_flag=False
            )
            sequences.append(sequence.cpu())
            representations.append(diffusion_rep.cpu())
            labels.append(torch.as_tensor(batch['next'].to_numpy(), dtype=torch.long))
    if not sequences:
        raise ValueError('No {} rows are available for score correction'.format(name))
    return {
        'sequence': torch.cat(sequences),
        'diffusion_rep': torch.cat(representations),
        'label': torch.cat(labels),
    }


def _sample_candidates(positive, item_count, negatives):
    if item_count < 2:
        raise ValueError('Negative sampling needs at least two target items')
    negative = torch.randint(1, item_count + 1,
                             (len(positive), negatives), device=positive.device)
    collision = negative.eq(positive.unsqueeze(1))
    negative = ((negative - 1 + collision.long()) % item_count) + 1
    return torch.cat([positive.unsqueeze(1), negative], dim=1)


def _evaluate(scorer, cached, args, batch_size):
    scorer.eval()
    metric_ks = sorted(set(args.metric_ks) | {10})
    totals = {prefix + '{}@{}'.format(metric, k): 0.0
              for prefix in ('Base_', 'Corrected_')
              for metric in ('HR', 'NDCG') for k in metric_ks}
    total_rows = 0
    with torch.no_grad():
        for start in range(0, len(cached['label']), batch_size):
            end = min(start + batch_size, len(cached['label']))
            sequence = cached['sequence'][start:end].to(args.device)
            diffusion_rep = cached['diffusion_rep'][start:end].to(args.device)
            labels = cached['label'][start:end].to(args.device).unsqueeze(1)
            corrected_rep, _, _ = scorer.query(sequence, diffusion_rep)
            for prefix, query in (('Base_', diffusion_rep),
                                  ('Corrected_', corrected_rep)):
                scores = mask_padding_item(scorer.scores(query))
                metrics = hrs_and_ndcgs_k(scores, labels, metric_ks)
                for metric, value in metrics.items():
                    totals[prefix + metric] += value * (end - start)
            total_rows += end - start
    return {key: value / total_rows for key, value in totals.items()}


def run_score_correction(model, train_data, val_data, test_data, args, logger):
    """Fit only the scoring head and compare it with the frozen original scorer."""
    if args.score_correction_epochs < 1 or args.score_correction_negatives < 1:
        raise ValueError('Score correction epochs and negatives must be positive')
    if args.score_correction_patience < 1 or args.score_correction_batch_size < 1:
        raise ValueError('Score correction patience and batch size must be positive')
    output_dir = Path(args.log_file) / '{}_{}'.format(args.s_dataset, args.t_dataset)
    stem = Path(args.checkpoint_name).stem + '_score_correction'
    report_path = output_dir / (stem + '.json')
    checkpoint_path = output_dir / (stem + '.pth')
    if report_path.exists() or checkpoint_path.exists():
        raise FileExistsError('Score correction output already exists: {}'.format(stem))
    core = model.module if isinstance(model, nn.DataParallel) else model
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    scorer = TargetResidualScorer(
        core.target_embeddings.weight,
        sparse_topk=args.score_correction_sparse_topk,
    ).to(args.device)
    cache_batch_size = args.score_correction_cache_batch_size
    logger.info('Caching label-free reverse outputs for score correction')
    train_cache = _cache_inference_representations(
        model, train_data, args, cache_batch_size, 'train')
    val_cache = _cache_inference_representations(
        model, val_data, args, cache_batch_size, 'val')

    optimizer = torch.optim.Adam(scorer.parameters(), lr=args.score_correction_lr)
    best_state = None
    best_ndcg = -math.inf
    best_epoch = 0
    bad_epochs = 0
    for epoch in range(1, args.score_correction_epochs + 1):
        scorer.train()
        order = torch.randperm(len(train_cache['label']))
        losses = []
        for indices in order.split(args.score_correction_batch_size):
            sequence = train_cache['sequence'][indices].to(args.device)
            diffusion_rep = train_cache['diffusion_rep'][indices].to(args.device)
            positive = train_cache['label'][indices].to(args.device)
            candidate_ids = _sample_candidates(
                positive, scorer.item_vectors.size(0) - 1,
                args.score_correction_negatives)
            corrected_rep, target_rep, reconstruction = scorer.query(
                sequence, diffusion_rep)
            corrected_scores = scorer.scores(corrected_rep, candidate_ids)
            target_scores = scorer.scores(target_rep, candidate_ids)
            label_index = torch.zeros(len(indices), dtype=torch.long,
                                      device=args.device)
            loss = (F.cross_entropy(corrected_scores, label_index)
                    + 0.2 * F.cross_entropy(target_scores, label_index)
                    + 0.05 * reconstruction)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        validation = _evaluate(
            scorer, val_cache, args, args.score_correction_eval_batch_size)
        current = validation['Corrected_NDCG@10']
        logger.info('Score correction epoch %d loss %.6f validation %s',
                    epoch, float(np.mean(losses)), validation)
        if current > best_ndcg:
            best_ndcg = current
            best_epoch = epoch
            best_state = copy.deepcopy(scorer.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.score_correction_patience:
                break

    scorer.load_state_dict(best_state)
    validation = _evaluate(
        scorer, val_cache, args, args.score_correction_eval_batch_size)
    test_cache = _cache_inference_representations(
        model, test_data, args, cache_batch_size, 'test')
    test = _evaluate(
        scorer, test_cache, args, args.score_correction_eval_batch_size)
    report = {'best_epoch': best_epoch, 'validation': validation, 'test': test,
              'validation_ndcg10_gain': (validation['Corrected_NDCG@10']
                                         - validation['Base_NDCG@10']),
              'test_ndcg10_gain': (test['Corrected_NDCG@10']
                                   - test['Base_NDCG@10']),
              'validation_supports_correction':
                  validation['Corrected_NDCG@10'] > validation['Base_NDCG@10'],
              'score_mixing_weight': float(torch.sigmoid(scorer.mix_logit).detach())}

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    torch.save({'base_model': core.state_dict(), 'scorer': scorer.state_dict(),
                'args': vars(args), 'report': report}, checkpoint_path)
    print('Score correction comparison:', report)
    logger.info('Score correction comparison: %s', report)
    logger.info('Score correction checkpoint: %s', checkpoint_path)
    return report
