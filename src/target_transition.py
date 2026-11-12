"""Sample target-train prefix/next-item pairs without exposing later items."""

import torch


def sample_internal_prefix(sequence):
    """Return one right-aligned prefix and its next item for each eligible row.

    The model reads the last physical sequence position, so valid prefixes must
    end there. Rows with fewer than two observed items are excluded.
    """
    if sequence.ndim != 2:
        raise ValueError('sequence must have shape [batch, sequence_length]')
    width = sequence.size(1)
    lengths = (sequence > 0).sum(dim=1)
    eligible = lengths >= 2
    sequence = sequence[eligible]
    lengths = lengths[eligible]
    if sequence.size(0) == 0:
        return sequence, sequence.new_empty((0, 1))

    # c is uniform over the valid prefix lengths 1, ..., L - 1.
    cuts = (torch.rand(sequence.size(0), device=sequence.device)
            * (lengths - 1)).long() + 1
    positions = torch.arange(width, device=sequence.device).unsqueeze(0)
    source_positions = positions + cuts.unsqueeze(1) - lengths.unsqueeze(1)
    prefix_mask = positions >= (width - cuts).unsqueeze(1)
    prefix = sequence.gather(1, source_positions.clamp(0, width - 1))
    prefix = prefix.masked_fill(~prefix_mask, 0)
    labels = sequence.gather(1, (width - lengths + cuts).unsqueeze(1))
    return prefix, labels
