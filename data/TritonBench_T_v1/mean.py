import torch
from typing import Optional

def torch_mean(input_tensor: torch.Tensor, 
        dim: int, 
        keepdim: bool=False, 
        dtype: Optional[torch.dtype]=None, 
        out: Optional[torch.Tensor]=None) -> torch.Tensor:
    """
    Computes the mean value of each row (or over specified dimensions) of the input tensor.

    Args:
        input_tensor (Tensor): The input tensor.
        dim (int or tuple of ints): The dimension or dimensions to reduce.
        keepdim (bool, optional): Whether the output tensor retains the same dimensions as the input tensor.
        dtype (torch.dtype, optional): The desired data type of the returned tensor.
        out (Tensor, optional): The output tensor.

    Returns:
        Tensor: The mean value of the tensor along the specified dimension(s).
    """
    return torch.mean(input_tensor, dim, keepdim=keepdim, dtype=dtype, out=out)

##################################################################################################################################################


import torch
torch.manual_seed(42)

def test_mean():
    results = {}

    # Test case 1: Basic mean computation over a single dimension
    input_tensor1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_1"] = torch_mean(input_tensor1, dim=0)

    # Test case 2: Mean computation with keepdim=True
    input_tensor2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_2"] = torch_mean(input_tensor2, dim=1, keepdim=True)

    # Test case 3: Mean computation over multiple dimensions
    input_tensor3 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device='cuda')
    results["test_case_3"] = torch_mean(input_tensor3, dim=(0, 2))

    # Test case 4: Mean computation with dtype specified
    input_tensor4 = torch.tensor([[1, 2], [3, 4]], device='cuda', dtype=torch.int32)
    results["test_case_4"] = torch_mean(input_tensor4, dim=0, dtype=torch.float32)

    return results

test_results = test_mean()
print(test_results)