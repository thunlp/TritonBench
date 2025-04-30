import torch
from typing import Optional

def sqrt_tanh(
        input: torch.Tensor, 
        out: Optional[torch.Tensor]=None) -> torch.Tensor:
    """
    Computes the square root of the input tensor, then applies the tanh function element-wise.

    Args:
        input (torch.Tensor): The input tensor.
        out (torch.Tensor, optional): The output tensor.
    
    Returns:
        torch.Tensor: The result of applying the square root and tanh functions element-wise.
    """
    if out is None:
        out = torch.empty_like(input)
    out = torch.sqrt(input)
    out[input < 0] = float('nan')
    out = torch.tanh(out)
    return out

##################################################################################################################################################


import torch
torch.manual_seed(42)

def test_sqrt_tanh():
    results = {}

    # Test case 1: Positive values
    input1 = torch.tensor([4.0, 9.0, 16.0], device='cuda')
    results["test_case_1"] = sqrt_tanh(input1)

    # Test case 2: Negative values
    input2 = torch.tensor([-4.0, -9.0, -16.0], device='cuda')
    results["test_case_2"] = sqrt_tanh(input2)

    # Test case 3: Mixed values
    input3 = torch.tensor([4.0, -9.0, 16.0, -1.0], device='cuda')
    results["test_case_3"] = sqrt_tanh(input3)

    # Test case 4: Zero values
    input4 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_4"] = sqrt_tanh(input4)

    return results

test_results = test_sqrt_tanh()
print(test_results)