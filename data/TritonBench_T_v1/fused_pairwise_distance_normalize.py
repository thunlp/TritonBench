import torch
from typing import Optional
def fused_pairwise_distance_normalize(
        x1: torch.Tensor, 
        x2: torch.Tensor, 
        p_norm: float=2.0, 
        eps_norm: float=1e-12, 
        eps_distance: float=1e-06, 
        keepdim: bool=False, 
        out: Optional[torch.Tensor]=None) -> torch.Tensor:
    """
    Computes the pairwise distance between two input tensors `x1` and `x2` 
    after normalizing both tensors. Normalization is performed along the specified 
    dimension, followed by pairwise distance calculation.

    Args:
        x1 (Tensor): First input tensor.
        x2 (Tensor): Second input tensor.
        p_norm (float, optional): The exponent value in the norm for normalization. Default: 2.
        eps_norm (float, optional): Small value to avoid division by zero during normalization. Default: 1e-12.
        eps_distance (float, optional): Small value to avoid division by zero in distance calculation. Default: 1e-6.
        keepdim (bool, optional): If `True`, retains the last dimension in the output. Default: `False`.

    Returns:
        torch.Tensor: The normalized pairwise distance tensor.
    """
    norm_x1 = torch.norm(x1, p=p_norm, dim=-1, keepdim=True)
    norm_x2 = torch.norm(x2, p=p_norm, dim=-1, keepdim=True)
    norm_x1 = torch.max(norm_x1, torch.tensor(eps_norm, device=x1.device))
    norm_x2 = torch.max(norm_x2, torch.tensor(eps_norm, device=x2.device))
    x1_normalized = x1 / norm_x1
    x2_normalized = x2 / norm_x2
    diff = x1_normalized.unsqueeze(1) - x2_normalized.unsqueeze(0)
    distance = torch.norm(diff, p=p_norm, dim=-1)
    distance = torch.max(distance, torch.tensor(eps_distance, device=x1.device))
    if keepdim:
        return distance.unsqueeze(-1)
    if out is not None:
        out.copy_(distance)
        return out
    return distance

##################################################################################################################################################


import torch
torch.manual_seed(42)

def test_fused_pairwise_distance_normalize():
    results = {}

    # Test case 1: Basic functionality with default parameters
    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    x2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    results["test_case_1"] = fused_pairwise_distance_normalize(x1, x2)

    # Test case 2: Different p_norm value
    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    x2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    results["test_case_2"] = fused_pairwise_distance_normalize(x1, x2, p_norm=1.0)

    # Test case 3: Different eps_norm value
    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    x2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    results["test_case_3"] = fused_pairwise_distance_normalize(x1, x2, eps_norm=1e-10)

    # Test case 4: keepdim=True
    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    x2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    results["test_case_4"] = fused_pairwise_distance_normalize(x1, x2, keepdim=True)

    return results

test_results = test_fused_pairwise_distance_normalize()
print(test_results)