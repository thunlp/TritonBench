import torch
from typing import Optional

def mul(input: torch.Tensor, 
        other: torch.Tensor, 
        out: Optional[torch.Tensor]=None) -> torch.Tensor:
    """
    Multiplies the input tensor by another tensor or a number, supporting broadcasting to a common shape,
    type promotion, and integer, float, and complex inputs.

    Parameters:
    - input (torch.Tensor): The input tensor.
    - other (torch.Tensor or Number): The tensor or number to multiply input by.
    - out (torch.Tensor, optional): The output tensor.

    Returns:
    - torch.Tensor: The result of the multiplication.
    """
    return torch.mul(input, other, out=out)

##################################################################################################################################################


import torch
torch.manual_seed(42)

def test_mul():
    results = {}

    # Test case 1: Multiply two tensors with broadcasting
    input1 = torch.tensor([1, 2, 3], device='cuda')
    other1 = torch.tensor([[1], [2], [3]], device='cuda')
    results["test_case_1"] = mul(input1, other1)

    # Test case 2: Multiply tensor by a scalar
    input2 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other2 = 2.5
    results["test_case_2"] = mul(input2, other2)

    # Test case 3: Multiply complex tensors
    input3 = torch.tensor([1+2j, 3+4j], device='cuda')
    other3 = torch.tensor([5+6j, 7+8j], device='cuda')
    results["test_case_3"] = mul(input3, other3)

    # Test case 4: Multiply integer tensor by a float tensor
    input4 = torch.tensor([1, 2, 3], device='cuda')
    other4 = torch.tensor([0.5, 1.5, 2.5], device='cuda')
    results["test_case_4"] = mul(input4, other4)

    return results

test_results = test_mul()
print(test_results)