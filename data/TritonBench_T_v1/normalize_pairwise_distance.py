import torch
import torch.nn.functional as F

def normalize_pairwise_distance(
        x1: torch.Tensor, 
        x2: torch.Tensor, 
        p_distance: float=2.0, 
        eps_distance: float=1e-06, 
        keepdim: bool=False, 
        p_norm: float=2, 
        dim_norm: int=1, 
        eps_norm: float=1e-12) -> torch.Tensor:
    """
    Computes the pairwise distance between `x1` and `x2` using the specified norm, 
    then normalizes the resulting distances along the specified dimension.
    
    Args:
        x1 (torch.Tensor): The first input tensor.
        x2 (torch.Tensor): The second input tensor, must have the same shape as `x1`.
        p_distance (float): The norm degree for computing the pairwise distance. Default: 2.0.
        eps_distance (float): Small value to avoid division by zero in pairwise distance calculation. Default: 1e-6.
        keepdim (bool): Whether to keep the reduced dimensions in the output. Default: False.
        p_norm (float): The exponent value in the norm formulation for normalization. Default: 2.
        dim_norm (int): The dimension along which normalization is applied. Default: 1.
        eps_norm (float): Small value to avoid division by zero in normalization. Default: 1e-12.

    Returns:
        torch.Tensor: The normalized pairwise distance between `x1` and `x2`.
    """
    pairwise_distance = torch.norm(x1 - x2, p=p_distance, dim=-1, keepdim=keepdim)
    pairwise_distance = pairwise_distance + eps_distance
    normed_distance = pairwise_distance / torch.norm(pairwise_distance, p=p_norm, dim=dim_norm, keepdim=True).clamp(min=eps_norm)
    return normed_distance

##################################################################################################################################################


import torch
torch.manual_seed(42)

def test_normalize_pairwise_distance():
    results = {}
    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x2 = torch.tensor([[1.0, 2.5], [2.5, 4.0]])
    
    # Compute the normalized pairwise distance
    results["test_case_1"] = normalize_pairwise_distance(x1, x2, p_distance=2.0, dim_norm=0)
    # Normalize along a different dimension
    results["test_case_2"] = normalize_pairwise_distance(x1, x2, p_distance=1.0, dim_norm=0)

    return results

test_results = test_normalize_pairwise_distance()
print(test_results)