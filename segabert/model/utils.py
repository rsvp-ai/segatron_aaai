import torch
from torch import nn
from typing import Optional

def weighted_sum(matrix: torch.Tensor,attention: torch.Tensor) -> torch.Tensor:
    """
    Args:
        matrix ():
        attention ():
    """
    if attention.dim() == 2 and matrix.dim() == 3:
        return attention.unsqueeze(1).bmm(matrix).squeeze(1)
    if attention.dim() == 3 and matrix.dim() == 3:
        return attention.bmm(matrix)
    if matrix.dim() - 1 < attention.dim():
        expanded_size = list(matrix.size())
        for i in range(attention.dim() - matrix.dim() + 1):
            matrix = matrix.unsqueeze(1)
            expanded_size.insert(i + 1, attention.size(i + 1))
        matrix = matrix.expand(*expanded_size)
    intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
    return intermediate.sum(dim=-2)

def get_range_vector(size: int, device) -> torch.Tensor:
    """
    """
    return torch.arange(0, size, dtype=torch.long).to(device)

def flatten_and_batch_shift_indices(indices: torch.LongTensor,
                                    sequence_length: int) -> torch.Tensor:
    """``indices`` of size ``(batch_size, d_1, ..., d_n)`` indexes into dimension 2 of a target tensor,
    which has size ``(batch_size, sequence_length, embedding_size)``. This function returns a vector
    that correctly indexes into the flattened target. The sequence length of the target must be provided
    to compute the appropriate offset.
    Args:
        indices (torch.LongTensor):
    """
    if torch.max(indices) >= sequence_length or torch.min(indices) < 0:
        raise ValueError("All the elements should be in range (0, {}), but found ({}, {})".format(
            sequence_length - 1, torch.min(indices).item(), torch.max(indices).item()))
    offsets = get_range_vector(indices.size(0), indices.device) * sequence_length
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)

    # (batch_size, d_1, ..., d_n) + (batch_size, 1, ..., 1)
    offset_indices = indices + offsets

    # (batch_size * d_1 * ... * d_n)
    offset_indices = offset_indices.view(-1)
    return offset_indices

def batched_index_select(target: torch.Tensor,
                         indices: torch.LongTensor,
                         flattened_indices: Optional[torch.LongTensor] = None) -> torch.Tensor:
    """Select ``target`` of size ``(batch_size, sequence_length, embedding_size)`` with ``indices`` of
    size ``(batch_size, d_1, ***, d_n)``.
    Args:
        target (torch.Tensor): A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
    """
    if flattened_indices is None:
        flattened_indices = flatten_and_batch_shift_indices(indices, target.size(1))

    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.view(-1, target.size(-1))

    # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices.size()) + [target.size(-1)]

    # Shape: (batch_size, d_1, ..., d_n, embedding_size)
    selected_targets = flattened_selected.view(*selected_shape)
    return selected_targets

def masked_softmax(vector: torch.Tensor,
                                     mask: torch.Tensor,
                                     dim: int = -1,
                                     mask_fill_value: float = -1e32) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked. This performs a softmax on just the non-masked positions of ``vector``. Passing ``None``
    in for the mask is also acceptable, which is just the regular softmax.
    """
    if mask is None:
        result = torch.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        masked_vector = vector.masked_fill((1 - mask).bool(), mask_fill_value)
        result = torch.softmax(masked_vector, dim=dim)
    return result


class AverageSpanExtractor(nn.Module):
    def __init__(self):
        super(AverageSpanExtractor, self).__init__()

    def forward(self,
                sequence_tensor: torch.FloatTensor,
                span_indices: torch.LongTensor,
                sequence_mask: torch.LongTensor = None,
                span_indices_mask: torch.LongTensor = None) -> torch.FloatTensor:
        # Shape (batch_size, num_spans, 1)
        span_starts, span_ends = span_indices.split(1, dim=-1)

        span_ends = span_ends - 1

        span_widths = span_ends - span_starts

        max_batch_span_width = span_widths.max().item() + 1

        # sequence_tensor (batch, length, dim)
        # global_attention_logits = self._global_attention(sequence_tensor)
        global_average_logits = torch.ones(sequence_tensor.size()[:2] + (1,)).float().to(sequence_tensor.device)

        # Shape: (1, 1, max_batch_span_width)
        max_span_range_indices = get_range_vector(max_batch_span_width,sequence_tensor.device).view(1, 1, -1)
        span_mask = (max_span_range_indices <= span_widths).float()

        # (batch_size, num_spans, 1) - (1, 1, max_batch_span_width)
        raw_span_indices = span_ends - max_span_range_indices
        span_mask = span_mask * (raw_span_indices >= 0).float()
        span_indices = torch.relu(raw_span_indices.float()).long()

        flat_span_indices = flatten_and_batch_shift_indices(span_indices, sequence_tensor.size(1))

        span_embeddings = batched_index_select(sequence_tensor, span_indices, flat_span_indices)

        span_attention_logits = batched_index_select(global_average_logits,span_indices,flat_span_indices).squeeze(-1)

        span_attention_weights = masked_softmax(span_attention_logits, span_mask)

        attended_text_embeddings = weighted_sum(span_embeddings, span_attention_weights)

        if span_indices_mask is not None:
            return attended_text_embeddings * span_indices_mask.unsqueeze(-1).float()

        return attended_text_embeddings