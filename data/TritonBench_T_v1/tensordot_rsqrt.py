from typing import Tuple, List
import torch

def tensordot_rsqrt(a: torch.Tensor, b: torch.Tensor, 
        dims: Tuple[int, Tuple[List[int], List[int]], List[List[int]]]) -> torch.Tensor:
    """
    Perform a tensor contraction between two tensors a and b over specified dimensions and apply 
    reciprocal square root to the resulting tensor.

    Args:
        a (torch.Tensor): Left tensor to contract.
        b (torch.Tensor): Right tensor to contract.
        dims (int, Tuple[List[int], List[int]], or List[List[int]]): Dimensions for contraction, 
            as per `torch.tensordot`.

    Returns:
        torch.Tensor: The reciprocal square root of the tensordot product of tensors a and b.
    """
    result = torch.tensordot(a, b, dims=dims)
    return torch.rsqrt(result)

##################################################################################################################################################


import torch

def test_tensordot_rsqrt():
    results = {}

    # Test case 1: Simple contraction with scalar result
    a = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    b = torch.tensor([4.0, 5.0, 6.0], device='cuda')
    dims = 1
    results["test_case_1"] = tensordot_rsqrt(a, b, dims)

    # Test case 2: Contraction with matrices
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    dims = ([1], [0])
    results["test_case_2"] = tensordot_rsqrt(a, b, dims)

    # Test case 3: Higher-dimensional tensors
    a = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device='cuda')
    b = torch.tensor([[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]], device='cuda')
    dims = ([2], [1])
    results["test_case_3"] = tensordot_rsqrt(a, b, dims)

    # Test case 4: Different dimensions for contraction
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')
    b = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], device='cuda')
    dims = ([1], [0])
    results["test_case_4"] = tensordot_rsqrt(a, b, dims)

    return results

test_results = test_tensordot_rsqrt()
print(test_results)