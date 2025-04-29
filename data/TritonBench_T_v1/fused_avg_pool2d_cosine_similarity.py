import torch
import torch.nn.functional as F

def fused_avg_pool2d_cosine_similarity(
        x1: torch.Tensor, 
        x2: torch.Tensor, 
        kernel_size: int, 
        stride: int=None, 
        padding: int=0, 
        eps: float=1e-08
        ) -> torch.Tensor:
    """
    Computes the cosine similarity between `x1` and `x2` along the specified dimension (dim=1),
    adds a singleton dimension, and applies 2D average pooling.

    Args:
        x1 (torch.Tensor): First input tensor.
        x2 (torch.Tensor): Second input tensor.
        kernel_size (int): The size of the pooling kernel.
        stride (int, optional): The stride of the pooling operation. Defaults to None, which uses kernel_size.
        padding (int, optional): The padding to apply to the input. Defaults to 0.
        eps (float, optional): A small value to prevent division by zero in cosine similarity. Defaults to 1e-8.

    Returns:
        torch.Tensor: The result after applying cosine similarity and average pooling.
    """
    cosine_sim = F.cosine_similarity(x1, x2, dim=1, eps=eps)
    cosine_sim = cosine_sim.unsqueeze(1)
    if stride is None:
        stride = kernel_size
    pooled_result = F.avg_pool2d(cosine_sim, kernel_size=kernel_size, stride=stride, padding=padding)
    return pooled_result

##################################################################################################################################################


import torch

def test_fused_avg_pool2d_cosine_similarity():
    results = {}

    # Test case 1: Basic test with default stride and padding
    x1 = torch.randn(1, 3, 8, 8, device='cuda')
    x2 = torch.randn(1, 3, 8, 8, device='cuda')
    results["test_case_1"] = fused_avg_pool2d_cosine_similarity(x1, x2, kernel_size=2)

    # Test case 2: Test with specified stride
    x1 = torch.randn(1, 3, 8, 8, device='cuda')
    x2 = torch.randn(1, 3, 8, 8, device='cuda')
    results["test_case_2"] = fused_avg_pool2d_cosine_similarity(x1, x2, kernel_size=2, stride=1)

    # Test case 3: Test with specified padding
    x1 = torch.randn(1, 3, 8, 8, device='cuda')
    x2 = torch.randn(1, 3, 8, 8, device='cuda')
    results["test_case_3"] = fused_avg_pool2d_cosine_similarity(x1, x2, kernel_size=2, padding=1)

    # Test case 4: Test with different eps value
    x1 = torch.randn(1, 3, 8, 8, device='cuda')
    x2 = torch.randn(1, 3, 8, 8, device='cuda')
    results["test_case_4"] = fused_avg_pool2d_cosine_similarity(x1, x2, kernel_size=2, eps=1e-6)

    return results

test_results = test_fused_avg_pool2d_cosine_similarity()
print(test_results)