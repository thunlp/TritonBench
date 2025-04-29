from typing import Optional
import torch

def tanh(input_tensor: torch.Tensor, 
        out_tensor: Optional[torch.Tensor]=None) -> torch.Tensor:
    """
    This function calculates the hyperbolic tangent of each element of the input tensor.
    
    Args:
        input_tensor (torch.Tensor): The input tensor.
        out_tensor (torch.Tensor, optional): The output tensor. If provided, the result is stored in this tensor.

    Returns:
        torch.Tensor: A tensor containing the element-wise hyperbolic tangent of the input.
    """
    return torch.tanh(input_tensor, out=out_tensor)

##################################################################################################################################################


import torch

def test_tanh():
    results = {}

    # Test case 1: Basic test with a simple tensor
    input_tensor_1 = torch.tensor([0.0, 1.0, -1.0, 0.5, -0.5], device='cuda')
    results["test_case_1"] = tanh(input_tensor_1)

    # Test case 2: Test with a 2D tensor
    input_tensor_2 = torch.tensor([[0.0, 1.0], [-1.0, 0.5]], device='cuda')
    results["test_case_2"] = tanh(input_tensor_2)

    # Test case 3: Test with a larger tensor
    input_tensor_3 = torch.randn(100, 100, device='cuda')
    results["test_case_3"] = tanh(input_tensor_3)

    # Test case 4: Test with an empty tensor
    input_tensor_4 = torch.tensor([], device='cuda')
    results["test_case_4"] = tanh(input_tensor_4)

    return results

test_results = test_tanh()
print(test_results)