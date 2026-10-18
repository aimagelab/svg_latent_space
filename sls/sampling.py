import torch
import torch.nn.functional as F


def greedy(next_token_logits):
    """
    Greedy decoding: selects the token with the highest probability.
    """
    return torch.argmax(next_token_logits, dim=-1, keepdim=True)


def multinomial(next_token_logits, num_samples=1):
    """
    Sampling: samples tokens from the probability distribution.
    """
    probs = F.softmax(next_token_logits, dim=-1)
    return torch.multinomial(probs, num_samples=num_samples)


def top_p(next_token_logits, p=0.9):
    """
    Top-p (nucleus) sampling: samples tokens from the smallest set
    whose cumulative probability exceeds p.
    """
    probs = F.softmax(next_token_logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Create mask: keep tokens until cumulative prob exceeds p
    # Shift by 1 to keep the first token that exceeds p
    mask = cumulative_probs - sorted_probs <= p

    # Zero out probabilities outside the nucleus
    filtered_probs = sorted_probs.clone()
    filtered_probs[~mask] = 0.0

    # Normalize the filtered probabilities
    filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)

    # Sample from filtered distribution
    sampled_sorted_idx = torch.multinomial(filtered_probs, num_samples=1)

    # Map back to original indices
    return sorted_indices.gather(-1, sampled_sorted_idx)


def top_k(next_token_logits, k=5):
    """
    Top-k sampling: samples tokens from the top k highest probability tokens.
    """
    probs = F.softmax(next_token_logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(probs, k=k, dim=-1)

    # Normalize the top-k probabilities
    top_k_probs /= top_k_probs.sum(dim=-1, keepdim=True)
    el = torch.multinomial(top_k_probs, num_samples=1)
    # Map back to the original indices
    return top_k_indices.gather(-1, el)  #


generation_methods = {
    "greedy": greedy,
    "multinomial": multinomial,
    "top_p": top_p,
    "top_k": top_k,
}
