import torch

def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None):
    """
    Adds the `other` tensor, scaled by `alpha`, to the `input` tensor and computes the mean value
    along the specified dimension(s).
    
    Parameters:
        input (Tensor): The input tensor.
        other (Tensor or Number): The tensor or number to add to input.
        dim (int or tuple of ints, optional): The dimension(s) to reduce. Default: None.
        alpha (Number, optional): The multiplier for `other`. Default: 1.
        keepdim (bool, optional): Whether the output tensor has dim retained or not. Default: False.
        dtype (torch.dtype, optional): The desired data type of the returned tensor. Default: None.
        out (Tensor, optional): The output tensor.

    Returns:
        Tensor: A tensor containing the mean of the result after addition and scaling.
    """
    if isinstance(other, (int, float)):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    result = input + alpha * other
    mean_result = result.mean(dim=dim, keepdim=keepdim, dtype=dtype)
    return mean_result

##################################################################################################################################################


import torch

def test_add_mean():
    results = {}

    # Test case 1: Basic addition and mean with default alpha
    input1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other1 = torch.tensor([0.5, 0.5, 0.5], device='cuda')
    results["test_case_1"] = add_mean(input1, other1)

    # Test case 2: Addition with scalar other and non-default alpha
    input2 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other2 = 0.5
    results["test_case_2"] = add_mean(input2, other2, alpha=2)

    # Test case 3: Addition with mean along a specific dimension
    input3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    other3 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    results["test_case_3"] = add_mean(input3, other3, dim=0)

    # Test case 4: Addition with mean and keepdim=True
    input4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    other4 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    results["test_case_4"] = add_mean(input4, other4, dim=1, keepdim=True)

    return results

test_results = test_add_mean()
