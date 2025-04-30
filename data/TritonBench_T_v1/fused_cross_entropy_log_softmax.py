import torch
import torch.nn.functional as F

def fused_cross_entropy_log_softmax(
        input: torch.Tensor, 
        target: torch.Tensor, 
        dim: int=1, 
        weight: torch.Tensor=None, 
        ignore_index: int=-100, 
        reduction: str='mean', 
        label_smoothing: float=0.0) -> torch.Tensor:
    """
    Computes the cross entropy loss with log softmax applied to the input logits.
    
    Args:
        input (Tensor): Input tensor of logits, where softmax will be computed along `dim`.
        target (Tensor): Ground truth class indices or probabilities.
        dim (int, optional): Dimension along which to compute log softmax. Default is 1.
        weight (Tensor, optional): Manual rescaling weight for each class.
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input gradient. Default: -100.
        reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default: 'mean'.
        label_smoothing (float, optional): Specifies the amount of smoothing to be applied, where 0.0 means no smoothing. Default: 0.0.

    Returns:
        Tensor: The computed loss.
    """
    log_probs = F.log_softmax(input, dim=dim)
    loss = F.cross_entropy(log_probs, target, weight=weight, ignore_index=ignore_index, reduction=reduction, label_smoothing=label_smoothing)
    return loss

##################################################################################################################################################


import torch
torch.manual_seed(42)

def test_fused_cross_entropy_log_softmax():
    results = {}
    
    # Test case 1: Basic test with default parameters
    input = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], device='cuda')
    target = torch.tensor([2, 1], device='cuda')
    results["test_case_1"] = fused_cross_entropy_log_softmax(input, target)
    
    # Test case 2: Test with label smoothing
    input = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], device='cuda')
    target = torch.tensor([2, 1], device='cuda')
    results["test_case_2"] = fused_cross_entropy_log_softmax(input, target, label_smoothing=0.1)
    
    # Test case 3: Test with weight
    input = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], device='cuda')
    target = torch.tensor([2, 1], device='cuda')
    weight = torch.tensor([1.0, 0.5, 2.0], device='cuda')
    results["test_case_3"] = fused_cross_entropy_log_softmax(input, target, weight=weight)
    
    # Test case 4: Test with sum reduction
    input = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], device='cuda')
    target = torch.tensor([2, 1], device='cuda')
    results["test_case_4"] = fused_cross_entropy_log_softmax(input, target, reduction='sum')
    
    return results

test_results = test_fused_cross_entropy_log_softmax()
print(test_results)