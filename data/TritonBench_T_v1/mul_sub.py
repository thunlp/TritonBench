import torch
from typing import Optional

def mul_sub(input: torch.Tensor, 
        other_mul: torch.Tensor, 
        other_sub: torch.Tensor, 
        alpha: float=1, 
        out: Optional[torch.Tensor]=None) -> torch.Tensor:
    """
    Multiplies the input tensor by another tensor or number, then subtracts another tensor or number from the result,
    scaled by a given alpha. This operation is performed element-wise.

    Args:
        input (torch.Tensor): The input tensor to be multiplied.
        other_mul (torch.Tensor or Number): The tensor or number to multiply with `input`.
        other_sub (torch.Tensor or Number): The tensor or number to subtract from the multiplication result.
        alpha (Number, optional): The multiplier for :attr:`other_sub`. Default is 1.
        out (torch.Tensor, optional): The output tensor.

    Returns:
        torch.Tensor: The result of the operation.
    """
    result = input * other_mul - alpha * other_sub
    if out is not None:
        out.copy_(result)
        return out
    return result

##################################################################################################################################################


import torch
torch.manual_seed(42)

def test_mul_sub():
    results = {}

    # Test case 1: input, other_mul, other_sub are tensors
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other_mul_tensor = torch.tensor([0.5, 1.5, 2.5], device='cuda')
    other_sub_tensor = torch.tensor([0.1, 0.2, 0.3], device='cuda')
    results["test_case_1"] = mul_sub(input_tensor, other_mul_tensor, other_sub_tensor)

    # Test case 2: input is a tensor, other_mul is a number, other_sub is a tensor
    other_mul_number = 2.0
    results["test_case_2"] = mul_sub(input_tensor, other_mul_number, other_sub_tensor)

    # Test case 3: input is a tensor, other_mul is a tensor, other_sub is a number
    other_sub_number = 0.5
    results["test_case_3"] = mul_sub(input_tensor, other_mul_tensor, other_sub_number)

    # Test case 4: input, other_mul, other_sub are numbers
    input_number = 3.0
    results["test_case_4"] = mul_sub(input_number, other_mul_number, other_sub_number)

    return results

test_results = test_mul_sub()
print(test_results)