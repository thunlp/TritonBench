import torch
from typing import Optional
def ones_like(
        input: torch.Tensor, 
        dtype: Optional[torch.dtype]=None, 
        layout: Optional[torch.layout]=None, 
        device: Optional[torch.device]=None, 
        requires_grad: bool=False, 
        memory_format: torch.memory_format=torch.preserve_format) -> torch.Tensor:
    """
    Returns a tensor filled with the scalar value 1, with the same size as the input tensor.
    It mirrors the properties of the input tensor unless specified otherwise.

    Args:
        input (torch.Tensor): The input tensor whose shape determines the output tensor's size.
        dtype (torch.dtype, optional): The desired data type of the returned tensor. Default is None.
        layout (torch.layout, optional): The desired layout of the returned tensor. Default is None.
        device (torch.device, optional): The desired device of the returned tensor. Default is None.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default is False.
        memory_format (torch.memory_format, optional): The desired memory format of the returned tensor. Default is torch.preserve_format.

    Returns:
        torch.Tensor: A tensor of the same size as the input, filled with ones.
    """
    return torch.ones(input.size(), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, memory_format=memory_format)

##################################################################################################################################################


import torch
torch.manual_seed(42)

def test_ones_like():
    results = {}

    # Test case 1: Basic test with default parameters
    input_tensor = torch.randn(2, 3, device='cuda')
    results["test_case_1"] = ones_like(input_tensor)

    # Test case 2: Test with a different dtype
    input_tensor = torch.randn(2, 3, device='cuda')
    results["test_case_2"] = ones_like(input_tensor, dtype=torch.float64)

    # Test case 3: Test with requires_grad=True
    input_tensor = torch.randn(2, 3, device='cuda')
    results["test_case_3"] = ones_like(input_tensor, requires_grad=True)

    # Test case 4: Test with a different device
    input_tensor = torch.randn(2, 3, device='cuda')
    results["test_case_4"] = ones_like(input_tensor, device='cuda')

    return results

test_results = test_ones_like()
print(test_results)