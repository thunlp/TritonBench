import torch
from typing import Optional
def i0(input_tensor: torch.Tensor, out: Optional[torch.Tensor]=None) -> torch.Tensor:
    """
    Computes the zero-order modified Bessel function of the first kind (I_0) for each element in the input tensor.

    Args:
        input_tensor (Tensor): The input tensor.
        out (Tensor, optional): The output tensor. If provided, the result will be saved to this tensor.
    
    Returns:
        Tensor: The result of applying the I_0 function to each element in the input tensor.
    """
    return torch.special.i0(input_tensor, out=out)

##################################################################################################################################################


import torch
torch.manual_seed(42)

def test_i0():
    results = {}

    # Test case 1: Simple tensor on GPU
    input_tensor_1 = torch.tensor([0.0, 1.0, 2.0], device='cuda')
    results["test_case_1"] = i0(input_tensor_1)

    # Test case 2: Larger tensor with negative values on GPU
    input_tensor_2 = torch.tensor([-1.0, -2.0, 3.0, 4.0], device='cuda')
    results["test_case_2"] = i0(input_tensor_2)

    # Test case 3: Tensor with mixed positive and negative values on GPU
    input_tensor_3 = torch.tensor([-3.0, 0.0, 3.0], device='cuda')
    results["test_case_3"] = i0(input_tensor_3)

    # Test case 4: Tensor with fractional values on GPU
    input_tensor_4 = torch.tensor([0.5, 1.5, 2.5], device='cuda')
    results["test_case_4"] = i0(input_tensor_4)

    return results

test_results = test_i0()
print(test_results)