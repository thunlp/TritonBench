from typing import Optional
import torch

def trunc(input: torch.Tensor, out: Optional[torch.Tensor]=None) -> torch.Tensor:
    """
    Truncate the elements of the input tensor to integers.

    Args:
        input (torch.Tensor): The input tensor containing floating-point values.
        out (torch.Tensor, optional): The output tensor. Default is None.

    Returns:
        torch.Tensor: A new tensor with the truncated integer values of the input tensor.
    """
    return torch.trunc(input, out=out)

##################################################################################################################################################


import torch

def test_trunc():
    results = {}

    # Test case 1: Simple tensor with positive and negative floats
    input1 = torch.tensor([1.5, -2.7, 3.3, -4.8], device='cuda')
    results["test_case_1"] = trunc(input1)

    # Test case 2: Tensor with zero and positive floats
    input2 = torch.tensor([0.0, 2.9, 5.1], device='cuda')
    results["test_case_2"] = trunc(input2)

    # Test case 3: Tensor with large positive and negative floats
    input3 = torch.tensor([12345.678, -98765.432], device='cuda')
    results["test_case_3"] = trunc(input3)

    # Test case 4: Tensor with mixed positive, negative, and zero floats
    input4 = torch.tensor([-0.1, 0.0, 0.1, -1.9, 1.9], device='cuda')
    results["test_case_4"] = trunc(input4)

    return results

test_results = test_trunc()
print(test_results)