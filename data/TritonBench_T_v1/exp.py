import torch
import math

def exp(input_tensor, out=None):
    """
    This function calculates the exponential of each element in the input tensor.

    Args:
    - input_tensor (Tensor): The input tensor containing the values to apply the exponential function on.
    - out (Tensor, optional): The output tensor where the result will be stored.

    Returns:
    - Tensor: A new tensor with the exponential of the elements of the input tensor.
    """
    return torch.exp(input_tensor, out=out)

##################################################################################################################################################


import torch

def test_exp():
    results = {}

    # Test case 1: Basic test with a simple tensor
    input_tensor_1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_1"] = exp(input_tensor_1)

    # Test case 2: Test with a tensor containing negative values
    input_tensor_2 = torch.tensor([-1.0, -2.0, -3.0], device='cuda')
    results["test_case_2"] = exp(input_tensor_2)

    # Test case 3: Test with a tensor containing zero
    input_tensor_3 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_3"] = exp(input_tensor_3)

    # Test case 4: Test with a larger tensor
    input_tensor_4 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
    results["test_case_4"] = exp(input_tensor_4)

    return results

test_results = test_exp()
