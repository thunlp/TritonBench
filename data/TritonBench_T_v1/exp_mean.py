import torch

def exp_mean(input: torch.Tensor, dim=None, keepdim=False, dtype=None, out=None) -> torch.Tensor:
    """
    Apply the exponential function to each element in the input tensor
    and compute the mean value of the result along the specified dimension
    or over all elements if no dimension is specified.
    
    Args:
        input (Tensor): Input tensor.
        dim (int, tuple of ints, optional): The dimension or dimensions along which to compute the mean. 
            If None, computes the mean over all elements in the input tensor.
        keepdim (bool, optional): Whether to retain the reduced dimensions in the result tensor.
        dtype (torch.dtype, optional): The desired data type of the returned tensor.
        out (Tensor, optional): A tensor to store the result.
    
    Returns:
        Tensor: The mean of the exponentiated values.
    """
    if dtype is not None:
        input = input.to(dtype)
    return_value = torch.exp(input).mean(dim=dim, keepdim=keepdim)
    if out is not None:
        out.copy_(return_value)
        return out
    return return_value

##################################################################################################################################################


import torch
torch.manual_seed(42)

def test_exp_mean():
    results = {}

    # Test case 1: Basic test with a 1D tensor on GPU
    input_tensor_1d = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_1"] = exp_mean(input_tensor_1d)

    # Test case 2: 2D tensor with dim specified
    input_tensor_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_2"] = exp_mean(input_tensor_2d, dim=0)

    # Test case 3: 2D tensor with keepdim=True
    results["test_case_3"] = exp_mean(input_tensor_2d, dim=1, keepdim=True)

    # Test case 4: 3D tensor with no dim specified (mean over all elements)
    input_tensor_3d = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device='cuda')
    results["test_case_4"] = exp_mean(input_tensor_3d)

    return results

test_results = test_exp_mean()
print(test_results)