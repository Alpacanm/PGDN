"""Source-led prototype construction; the original MDR sampler is unchanged."""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def _train_batches(data, batch_size):
    for start in range(0, len(data), batch_size):
        yield data.iloc[start:start + batch_size]


def _history_representations(model, data, source, batch_size, device):
    """Use exactly the raw-history input and encoder used by legacy MDR queries."""
    table = model.souce_embeddings if source else model.target_embeddings
    memory = model.prototype_mdr
    chunks = []
    model.eval()
    with torch.no_grad():
        for rows in _train_batches(data, batch_size):
            sequence = torch.as_tensor(
                list(rows['seq'].values), dtype=torch.long, device=device)
            positions = torch.arange(sequence.size(1), device=device)
            embeddings = table(sequence) + model.position_embeddings(positions)[None, :, :]
            chunks.append(memory.encode(embeddings, (sequence > 0).float()))
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def _source_keys(source, count, iterations):
    """Spherical k-means++ followed by source-only centroid updates."""
    if len(source) < count:
        raise ValueError('Fewer source train histories than prototype candidates')
    selected = [int(torch.randint(len(source), (1,), device=source.device).item())]
    distances = (1.0 - source @ source[selected[0]]).clamp_min(0).square()
    for _ in range(1, count):
        if float(distances.sum()) > 1e-12:
            index = int(torch.multinomial(distances, 1).item())
        else:
            index = int(torch.randint(len(source), (1,), device=source.device).item())
        selected.append(index)
        new_distances = (1.0 - source @ source[index]).clamp_min(0).square()
        distances = torch.minimum(distances, new_distances)
        distances[selected] = 0
    keys = source[selected].clone()
    for _ in range(iterations):
        assignment = (source @ keys.T).argmax(dim=-1)
        sums = torch.zeros_like(keys)
        sums.index_add_(0, assignment, source)
        counts = torch.bincount(assignment, minlength=count)
        occupied = counts > 0
        keys[occupied] = F.normalize(sums[occupied], dim=-1)
    return keys


@torch.no_grad()
def _statistics(model, source, selected_target, keys, supported, args):
    memory = model.prototype_mdr
    source_assignment = (source @ keys.T).argmax(dim=-1)
    source_count = torch.bincount(source_assignment, minlength=memory.num_prototypes)
    source_sum = torch.zeros_like(memory.source_mean)
    source_square_sum = torch.zeros_like(memory.source_second_moment)
    source_sum.index_add_(0, source_assignment, source)
    source_square_sum.index_add_(0, source_assignment, source.square())
    mean = source_sum / source_count.clamp_min(1)[:, None]
    second = source_square_sum / source_count.clamp_min(1)[:, None]

    target_assignment = (selected_target @ keys.T).argmax(dim=-1)
    target_count = torch.bincount(target_assignment, minlength=memory.num_prototypes)
    # Target support controls publicness without changing source-derived keys.
    target_count = torch.where(supported, target_count, torch.zeros_like(target_count))
    target_count = torch.where(
        target_count >= args.prototype_min_target_support,
        target_count, torch.zeros_like(target_count))
    memory.prototypes.copy_(keys)
    memory.source_mean.copy_(mean)
    memory.source_second_moment.copy_(second)
    memory.source_usage.copy_(source_count.float() / source_count.sum().clamp_min(1))
    memory.target_usage.copy_(target_count.float() / target_count.sum().clamp_min(1))
    memory.source_seen.copy_((source_count >= args.prototype_min_source_support).float())
    return source_count, target_count


def build_source_guided_bank(model, source_train, target_train, args, logger):
    """Build the original bank with one explicitly selected correction mode."""
    correction_mode = getattr(args, 'prototype_correction_mode', 'none')
    if correction_mode not in ('key_rotation', 'none', 'preference_ridge'):
        raise ValueError('Unknown prototype correction mode: ' + correction_mode)
    if not 0 < args.prototype_target_fraction <= 1:
        raise ValueError('prototype_target_fraction must be in (0, 1]')
    if args.prototype_source_cluster_iters < 1:
        raise ValueError('Prototype cluster iterations must be positive')
    if not 0 <= args.prototype_target_correction <= 1:
        raise ValueError('prototype_target_correction must be in [0, 1]')
    if not 0 <= args.prototype_confidence_quantile <= 1:
        raise ValueError('prototype_confidence_quantile must be in [0, 1]')
    if not 0 <= args.prototype_source_similarity_quantile <= 1:
        raise ValueError('prototype_source_similarity_quantile must be in [0, 1]')
    if args.prototype_min_source_support < 1 or args.prototype_min_target_support < 1:
        raise ValueError('Prototype support thresholds must be positive')
    if len(source_train) == 0 or len(target_train) == 0:
        raise ValueError('Both source and target train data must be nonempty')

    target_rows = min(len(target_train), max(
        args.batch_size, math.ceil(len(target_train) * args.prototype_target_fraction)))
    # Fixed, unpaired target train subset used only for prototype direction correction.
    target_subset = target_train.sample(
        n=target_rows, replace=False, random_state=args.random_seed + 1701)

    source = _history_representations(
        model, source_train, True, args.batch_size, args.device)
    target = _history_representations(
        model, target_subset, False, args.batch_size, args.device)
    memory = model.prototype_mdr
    with torch.no_grad():
        memory.preference_delta.zero_()
        memory.preference_correction_active.fill_(False)
        source_keys = _source_keys(
            source, memory.num_prototypes, args.prototype_source_cluster_iters)
        source_best = (source @ source_keys.T).max(dim=-1).values
        source_floor = torch.quantile(
            source_best.float(), args.prototype_source_similarity_quantile)

        top = (target @ source_keys.T).topk(min(2, memory.num_prototypes), dim=-1)
        margin = (top.values[:, 0] - top.values[:, 1]
                  if memory.num_prototypes > 1 else torch.ones_like(top.values[:, 0]))
        margin_floor = torch.quantile(margin.float(), args.prototype_confidence_quantile)
        confident = ((top.values[:, 0] >= source_floor)
                     & (margin >= margin_floor)
                     & (margin > 1e-5))
        selected_target = target[confident]
        selected_assignment = top.indices[confident, 0]
        initial_target_count = torch.bincount(
            selected_assignment, minlength=memory.num_prototypes)
        initial_source_count = torch.bincount(
            (source @ source_keys.T).argmax(dim=-1),
            minlength=memory.num_prototypes)
        supported = ((initial_source_count >= args.prototype_min_source_support)
                     & (initial_target_count >= args.prototype_min_target_support))
        keys = source_keys.clone()
        if correction_mode == 'key_rotation' and args.prototype_target_correction > 0:
            target_sum = torch.zeros_like(source_keys)
            target_sum.index_add_(0, selected_assignment, selected_target)
            target_direction = F.normalize(target_sum, dim=-1)
            keys[supported] = F.normalize(
                (1.0 - args.prototype_target_correction) * source_keys[supported]
                + args.prototype_target_correction * target_direction[supported], dim=-1)
        rotation = torch.rad2deg(torch.acos(
            (source_keys * keys).sum(dim=-1).clamp(-1.0, 1.0)))
        source_count, target_count = _statistics(
            model, source, selected_target, keys, supported, args)
        preference_report = None
        if correction_mode == 'preference_ridge':
            from prototype_preference_correction import (next_item_preferences, fit_preference_delta,
                                                         supported_target_labels)
            fit_mask = supported_target_labels(
                target_subset, target_train, model.target_embeddings.num_embeddings)
            supervised_target = target_subset.iloc[fit_mask.nonzero().flatten().numpy()]
            source_future = next_item_preferences(
                model, source_train, True, args.batch_size, args.max_len, args.device)
            target_future = (next_item_preferences(
                model, supervised_target, False, args.batch_size, args.max_len, args.device)
                if len(supervised_target) else target.new_empty((0, target.size(1))))
            delta, preference_report = fit_preference_delta(
                source, source_future, target[fit_mask.to(target.device)], target_future, memory, supported,
                args.random_seed,
                source_ridge=getattr(args, 'prototype_preference_source_ridge', 0.01),
                target_ridges=getattr(args, 'prototype_preference_target_ridges',
                                      [0.001, 0.01, 0.1, 1.0, 10.0]))
            preference_report.update(original_target_subset_rows=len(target_subset),
                unsupported_target_labels_excluded=int((~fit_mask).sum()),
                target_label_history_coverage=float(fit_mask.float().mean()),
                missing_label_policy='exclude only from offline correction fit; never use zero/random preference targets')
            memory.preference_delta.copy_(delta)
            memory.preference_correction_active.fill_(preference_report['enabled'])
            logger.info('TRAIN-only prototype preference correction: %s', preference_report)
        public = memory.publicness()
        valid = (memory.source_seen > 0) & (public >= memory.min_public)
        source_share = source_count.float() / source_count.sum().clamp_min(1)
        source_effective = 1.0 / source_share.square().sum().clamp_min(1e-12)
        similarity = keys @ keys.T
        mask = ~torch.eye(memory.num_prototypes, dtype=torch.bool, device=keys.device)
        report = {
            'mode': 'source_guided',
            'correction_mode': correction_mode,
            'preference_correction': preference_report,
            'source_train_rows': len(source_train),
            'target_train_rows': len(target_train),
            'target_subset_rows': len(target_subset),
            'target_subset_random_state': args.random_seed + 1701,
            'target_fraction_requested': args.prototype_target_fraction,
            'target_embedding_preparation': False,
            'source_cluster_iters': args.prototype_source_cluster_iters,
            'target_correction_weight': (args.prototype_target_correction
                                         if correction_mode == 'key_rotation' else 0.0),
            'target_confident_rows': int(confident.sum()),
            'source_similarity_floor': float(source_floor),
            'target_margin_floor': float(margin_floor),
            'corrected_keys': (int(supported.sum()) if correction_mode == 'key_rotation'
                               and args.prototype_target_correction > 0 else
                               (preference_report or {}).get('corrected_keys', 0)),
            'retrievable_keys': int(valid.sum()),
            'source_effective_keys': float(source_effective),
            'max_source_share': float(source_share.max()),
            'max_key_similarity': float(similarity[mask].max()) if bool(mask.any()) else 0.0,
            'max_rotation_degrees': float(rotation.max()),
            'source_support': source_count.cpu().tolist(),
            'initial_target_support': initial_target_count.cpu().tolist(),
            'target_support': target_count.cpu().tolist(),
            'rotation_degrees': rotation.cpu().tolist(),
            'retrievable_mask': valid.cpu().tolist(),
        }
    report_path = (Path(args.log_file) / (args.s_dataset + '_' + args.t_dataset)
                   / 'source_guided_prototype_bank.json')
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    message = ('Source-guided bank: source={} target_subset={} confident={} '
               'corrected_keys={} retrievable_keys={} effective_keys={:.2f} '
               'max_source_share={:.4f} max_rotation={:.2f}deg report={}').format(
                   len(source_train), len(target_subset), report['target_confident_rows'],
                   report['corrected_keys'], report['retrievable_keys'],
                   report['source_effective_keys'], report['max_source_share'],
                   report['max_rotation_degrees'], report_path)
    print(message)
    logger.info(message)
    return report
