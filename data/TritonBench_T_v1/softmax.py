import torch
import torch.nn.functional as F


def softmax(input: torch.Tensor, dim: int, dtype: torch.dtype=None) -> torch.Tensor:
    """
    Apply softmax function to the input tensor along the specified dimension.
    The elements in the tensor will be scaled to the range [0, 1] and sum to 1 along the specified dimension.

    Args:
        input (torch.Tensor): The input tensor to apply softmax to.
        dim (int): The dimension along which softmax will be computed.
        dtype (torch.dtype, optional): The desired data type of the returned tensor. 
            If specified, the input tensor is casted to dtype before the operation is performed. 
            This is useful for preventing data type overflows. Default: None.

    Returns:
        torch.Tensor: The tensor with softmax applied.
    """
    return F.softmax(input, dim=dim, dtype=dtype)

##################################################################################################################################################


import torch

def test_softmax():
    results = {}
    
    # Test case 1: Basic test with default dtype
    input1 = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], device='cuda')
    results["test_case_1"] = softmax(input1, dim=1)
    
    # Test case 2: Test with different dimension
    input2 = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], device='cuda')
    results["test_case_2"] = softmax(input2, dim=0)
    
    # Test case 3: Test with specified dtype
    input3 = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], device='cuda')
    results["test_case_3"] = softmax(input3, dim=1, dtype=torch.float64)
    
    # Test case 4: Test with larger tensor
    input4 = torch.randn(100, 100, device='cuda')
    results["test_case_4"] = softmax(input4, dim=1)
    
    return results

test_results = test_softmax()
print(test_results)