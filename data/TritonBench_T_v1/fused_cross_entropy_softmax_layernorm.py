import torch
import torch.nn.functional as F

def fused_cross_entropy_softmax_layernorm(
        logits: torch.Tensor, 
        targets: torch.Tensor, 
        normalized_shape: tuple[int], 
        weight: torch.Tensor=None, 
        ignore_index: int=-100, 
        reduction: str='mean', 
        label_smoothing: float=0.0, 
        eps: float=1e-05, 
        *, out: torch.Tensor=None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the cross entropy loss with softmax applied to the logits and layer normalization applied to the probabilities.
    
    Args:
        logits (torch.Tensor): The logits tensor.
        targets (torch.Tensor): The target tensor.
        normalized_shape (tuple[int]): The shape of the layer normalization.
        weight (torch.Tensor, optional): The weight tensor.
        ignore_index (int, optional): The ignore index.
        reduction (str, optional): The reduction method.
        label_smoothing (float, optional): The label smoothing.
        eps (float, optional): The epsilon.
        out (torch.Tensor, optional): The output tensor.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The loss and the output tensor.
    """
    loss = F.cross_entropy(logits, targets, weight=weight, ignore_index=ignore_index, reduction=reduction, label_smoothing=label_smoothing)
    probabilities = F.softmax(logits, dim=-1)
    output = F.layer_norm(probabilities, normalized_shape=(normalized_shape,), weight=None, bias=None, eps=eps)
    if out is not None:
        out.copy_(output)
        return (loss, out)
    return (loss, output)

##################################################################################################################################################


import torch

def test_fused_cross_entropy_softmax_layernorm():
    results = {}

    # Test case 1: Basic functionality with default parameters
    logits = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], device='cuda')
    targets = torch.tensor([2, 1], device='cuda')
    normalized_shape = 3
    loss, output = fused_cross_entropy_softmax_layernorm(logits, targets, normalized_shape)
    results["test_case_1"] = (loss.item(), output.cpu().numpy())

    # Test case 2: With weight parameter
    weight = torch.tensor([0.1, 0.2, 0.3], device='cuda')
    loss, output = fused_cross_entropy_softmax_layernorm(logits, targets, normalized_shape, weight=weight)
    results["test_case_2"] = (loss.item(), output.cpu().numpy())

    # Test case 3: With ignore_index parameter
    targets_ignore = torch.tensor([2, -100], device='cuda')
    loss, output = fused_cross_entropy_softmax_layernorm(logits, targets_ignore, normalized_shape, ignore_index=-100)
    results["test_case_3"] = (loss.item(), output.cpu().numpy())

    # Test case 4: With label_smoothing parameter
    loss, output = fused_cross_entropy_softmax_layernorm(logits, targets, normalized_shape, label_smoothing=0.1)
    results["test_case_4"] = (loss.item(), output.cpu().numpy())

    return results

test_results = test_fused_cross_entropy_softmax_layernorm()
print(test_results)