import torch
from typing import Optional

def logsumexp(
        input: torch.Tensor, 
        dim: int, 
        keepdim: bool=False, 
        out: Optional[torch.Tensor]=None) -> torch.Tensor:
    """
    Computes the logarithm of the sum of exponentials of input elements along the specified dimension.

    Args:
        input (Tensor): Input tensor.
        dim (int): Dimension along which to compute the logsumexp.
        keepdim (bool, optional): Whether to retain the reduced dimension. Default: False.
        out (Tensor, optional): The output tensor. Default: None.

    Returns:
        Tensor: The computed logsumexp values along the specified dimension.
    """
    return torch.logsumexp(input, dim, keepdim)

##################################################################################################################################################


import torch

def test_logsumexp():
    results = {}

    # Test case 1: Basic test with a 2D tensor on GPU
    input_tensor_1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_1"] = logsumexp(input_tensor_1, dim=0)

    # Test case 2: Test with keepdim=True
    input_tensor_2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_2"] = logsumexp(input_tensor_2, dim=1, keepdim=True)

    # Test case 3: Test with a 3D tensor
    input_tensor_3 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device='cuda')
    results["test_case_3"] = logsumexp(input_tensor_3, dim=2)

    # Test case 4: Test with a negative dimension
    input_tensor_4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_4"] = logsumexp(input_tensor_4, dim=-1)

    return results

test_results = test_logsumexp()
print(test_results)