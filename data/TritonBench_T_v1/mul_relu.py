import torch
import torch.nn.functional as F
from typing import Optional
def mul_relu(
        input: torch.Tensor, 
        other: torch.Tensor, 
        inplace: bool=False, 
        out: Optional[torch.Tensor]=None) -> torch.Tensor:
    """
    This function performs element-wise multiplication of two inputs, input and other, 
    and then applies the Rectified Linear Unit (ReLU) function to the result, 
    which replaces all negative values with zero.

    Args:
        input (Tensor): The input tensor to be multiplied.
        other (Tensor or Number): The tensor or number to multiply with input.
        inplace (bool, optional): If True, modifies input in-place, if possible. Default is False.
        out (Tensor, optional): The output tensor.

    Returns:
        Tensor: A tensor with the element-wise multiplication result followed by ReLU activation.
    """
    result = torch.mul(input, other)
    out_relu = F.relu(result, inplace=inplace)
    if out is not None:
        out.copy_(out_relu)
    return out_relu

##################################################################################################################################################


import torch

def test_mul_relu():
    results = {}

    # Test case 1: Basic multiplication and ReLU with two tensors
    input1 = torch.tensor([-1.0, 2.0, -3.0, 4.0], device='cuda')
    other1 = torch.tensor([1.0, -1.0, 1.0, -1.0], device='cuda')
    results["test_case_1"] = mul_relu(input1, other1)

    # Test case 2: Multiplication with a scalar
    input2 = torch.tensor([-1.0, 2.0, -3.0, 4.0], device='cuda')
    other2 = 2.0
    results["test_case_2"] = mul_relu(input2, other2)

    # Test case 3: In-place operation
    input3 = torch.tensor([-1.0, 2.0, -3.0, 4.0], device='cuda')
    other3 = torch.tensor([1.0, -1.0, 1.0, -1.0], device='cuda')
    results["test_case_3"] = mul_relu(input3, other3, inplace=True)

    # Test case 4: Multiplication with a different shaped tensor
    input4 = torch.tensor([[-1.0, 2.0], [-3.0, 4.0]], device='cuda')
    other4 = torch.tensor([[1.0, -1.0], [1.0, -1.0]], device='cuda')
    results["test_case_4"] = mul_relu(input4, other4)

    return results

test_results = test_mul_relu()
print(test_results)