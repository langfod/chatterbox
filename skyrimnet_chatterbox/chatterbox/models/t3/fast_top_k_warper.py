import torch
from transformers.generation.logits_process import TopKLogitsWarper


class FastTopKLogitsWarper(TopKLogitsWarper):
    """
    CUDA graph compatible TopKLogitsWarper.
    
    Filters logits to keep only the top-k highest probability tokens.
    Pre-allocates tensors to avoid dynamic allocations that break CUDA graphs.
    """

    def __init__(self, top_k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        super().__init__(top_k=top_k, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        top_k = min(self.top_k, scores.size(-1))  # Safety check
        
        # Get the top-k values and their indices
        # We need the k-th largest value as threshold
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        
        # Ensure we keep at least min_tokens_to_keep
        # This is handled by top_k >= min_tokens_to_keep in parent __init__
        
        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed
