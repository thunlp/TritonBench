import torch
from typing import Sequence, Union, Optional

def fused_hstack_div(
        tensors: Sequence[torch.Tensor], 
        divisor: Union[torch.Tensor, Union[int, float]], 
        *, 
        rounding_mode: Optional[str]=None, 
        out: Optional[torch.Tensor]=None) -> torch.Tensor:
    """
    Performs a fused operation combining horizontal stacking (hstack) and element-wise division.

    Args:
        tensors (sequence of Tensors): Sequence of tensors to be horizontally stacked.
                                        The tensors must have compatible shapes for stacking.
        divisor (Tensor or number): The tensor or number to divide the stacked tensor by.
                                    Must be broadcastable to the shape of the stacked tensor.
        rounding_mode (str, optional): Type of rounding applied to the result. Options:
                                       'None', 'trunc', 'floor'. Default: None.
        out (Tensor, optional): Output tensor. Ignored if None. Default: None.

    Returns:
        Tensor: The result of stacking the tensors horizontally and dividing element-wise by the divisor.
    """
    X = torch.hstack(tensors)
    Y = torch.div(X, divisor, rounding_mode=rounding_mode)
    if out is not None:
        out.copy_(Y)
        return out
    return Y

##################################################################################################################################################


import torch

def test_fused_hstack_div():
    results = {}

    # Test case 1: Basic functionality with two tensors and a scalar divisor
    tensors1 = [torch.tensor([1, 2], device='cuda'), torch.tensor([3, 4], device='cuda')]
    divisor1 = 2
    results["test_case_1"] = fused_hstack_div(tensors1, divisor1)

    # Test case 3: Using rounding_mode='floor'
    tensors3 = [torch.tensor([1.5, 2.5], device='cuda'), torch.tensor([3.5, 4.5], device='cuda')]
    divisor3 = 2
    results["test_case_3"] = fused_hstack_div(tensors3, divisor3, rounding_mode='floor')

    # Test case 4: Using rounding_mode='trunc'
    tensors4 = [torch.tensor([1.5, 2.5], device='cuda'), torch.tensor([3.5, 4.5], device='cuda')]
    divisor4 = 2
    results["test_case_4"] = fused_hstack_div(tensors4, divisor4, rounding_mode='trunc')

    return results

test_results = test_fused_hstack_div()
print(test_results)