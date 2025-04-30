import torch
import torch.nn.functional as F
from typing import Optional

def softmax_mul(
        input: torch.Tensor, 
        other: torch.Tensor, 
        dim: int, 
        dtype: Optional[torch.dtype]=None, 
        out: Optional[torch.Tensor]=None) -> torch.Tensor:    
    """
    Applies the softmax function to the input tensor along the specified dimension,
    then multiplies the result with another tensor element-wise.

    Args:
        input (torch.Tensor): The input tensor.
        other (torch.Tensor): The tensor to multiply with.
        dim (int): The dimension along which softmax will be computed.
        dtype (:class:`torch.dtype`, optional): The desired data type of the returned tensor.
                                                If specified, the input tensor is cast to :attr:`dtype`
                                                before the operation is performed. Default: None.
        out (torch.Tensor, optional): The output tensor.

    Returns:
        torch.Tensor: The result of applying softmax and multiplication.
    """
    softmaxed = F.softmax(input, dim=dim, dtype=dtype)
    if isinstance(other, torch.Tensor):
        result = softmaxed * other
    else:
        result = softmaxed * other
    if out is not None:
        out.copy_(result)
        return out
    return result

##################################################################################################################################################


import torch
torch.manual_seed(42)
def test_softmax_mul():
    results = {}
    
    # Test case 1: Basic test with two tensors
    input1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    other1 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    results["test_case_1"] = softmax_mul(input1, other1, dim=1)
    
    # Test case 2: Test with scalar multiplication
    input2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    other2 = 0.5
    results["test_case_2"] = softmax_mul(input2, other2, dim=1)
    
    # Test case 3: Test with different dtype
    input3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    other3 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    results["test_case_3"] = softmax_mul(input3, other3, dim=1, dtype=torch.float64)
    
    # Test case 4: Test with out parameter
    input4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    other4 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    out4 = torch.empty_like(input4)
    results["test_case_4"] = softmax_mul(input4, other4, dim=1, out=out4)
    
    return results

test_results = test_softmax_mul()
print(test_results)