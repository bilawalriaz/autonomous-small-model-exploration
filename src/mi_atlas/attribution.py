"""Attribution methods (gradient-based)."""

import torch


def compute_gradient_attribution(
    model,
    input_ids: torch.Tensor,
    target_token_id: int,
) -> torch.Tensor:
    """Compute gradient of target logit w.r.t. input embeddings.

    Simple integrated gradients approximation.
    """
    embeddings = model.get_input_embeddings()(input_ids)
    embeddings.retain_grad()

    # Forward pass
    outputs = model(inputs_embeds=embeddings)
    logits = outputs.logits

    # Backprop from target logit at last position
    target_logit = logits[0, -1, target_token_id]
    target_logit.backward()

    # Attribution = gradient * input
    if embeddings.grad is not None:
        attribution = (embeddings.grad * embeddings).sum(dim=-1)
        return attribution.detach().cpu()

    return torch.zeros(input_ids.shape)


def attribution_patching_score(
    clean_attribution: torch.Tensor,
    corrupt_attribution: torch.Tensor,
) -> torch.Tensor:
    """Approximate activation patching using attributions."""
    return clean_attribution - corrupt_attribution
