import torch
import torch.nn.functional as F

def fused_pairwise_distance_adaptive_avg_pool2d(x1: torch.Tensor, x2: torch.Tensor, output_size: int or tuple, p: float=2.0, eps: float=1e-06, keepdim: bool=False) -> torch.Tensor:
    """
    This function applies adaptive average pooling to the input tensors `x1` and `x2` to resize them
    to the specified `output_size`, and then computes the pairwise distance between the pooled outputs.

    Args:
        x1 (Tensor): First input tensor for adaptive average pooling and distance calculation.
        x2 (Tensor): Second input tensor for adaptive average pooling and distance calculation.
        output_size (int or tuple): The target output size for the adaptive average pooling.
        p (float, optional): The norm degree for pairwise distance calculation. Default: 2.0
        eps (float, optional): Small value to avoid division by zero in pairwise distance. Default: 1e-6
        keepdim (bool, optional): Whether to keep the reduced dimension. Default: False

    Returns:
        Tensor: A tensor containing the pairwise distance between the pooled tensors.
    """
    pooled_x1 = F.adaptive_avg_pool2d(x1, output_size)
    pooled_x2 = F.adaptive_avg_pool2d(x2, output_size)
    diff = pooled_x1 - pooled_x2
    dist = torch.norm(diff, p=p, dim=(1, 2, 3), keepdim=keepdim) + eps
    return dist

##################################################################################################################################################


import torch
import torch.nn.functional as F

# def fused_pairwise_distance_adaptive_avg_pool2d(x1: torch.Tensor, x2: torch.Tensor, output_size: int or tuple, p: float=2.0, eps: float=1e-06, keepdim: bool=False) -> torch.Tensor:
#     pooled_x1 = F.adaptive_avg_pool2d(x1, output_size)
#     pooled_x2 = F.adaptive_avg_pool2d(x2, output_size)
#     diff = pooled_x1 - pooled_x2
#     dist = torch.norm(diff, p=p, dim=(1, 2, 3), keepdim=keepdim) + eps
#     return dist

def test_fused_pairwise_distance_adaptive_avg_pool2d():
    results = {}
    
    # Test case 1: Basic test with default parameters
    x1 = torch.rand((2, 3, 32, 32), device='cuda')
    x2 = torch.rand((2, 3, 32, 32), device='cuda')
    results["test_case_1"] = fused_pairwise_distance_adaptive_avg_pool2d(x1, x2, output_size=(8, 8))

    # Test case 2: Different output size
    x1 = torch.rand((2, 3, 64, 64), device='cuda')
    x2 = torch.rand((2, 3, 64, 64), device='cuda')
    results["test_case_2"] = fused_pairwise_distance_adaptive_avg_pool2d(x1, x2, output_size=(16, 16))

    # Test case 3: Different norm degree
    x1 = torch.rand((2, 3, 32, 32), device='cuda')
    x2 = torch.rand((2, 3, 32, 32), device='cuda')
    results["test_case_3"] = fused_pairwise_distance_adaptive_avg_pool2d(x1, x2, output_size=(8, 8), p=1.0)

    # Test case 4: Keep dimension
    x1 = torch.rand((2, 3, 32, 32), device='cuda')
    x2 = torch.rand((2, 3, 32, 32), device='cuda')
    results["test_case_4"] = fused_pairwise_distance_adaptive_avg_pool2d(x1, x2, output_size=(8, 8), keepdim=True)

    return results

test_results = test_fused_pairwise_distance_adaptive_avg_pool2d()
