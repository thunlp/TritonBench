import torch

def ones_like(input, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format):
    """
    Returns a tensor filled with the scalar value 1, with the same size as the input tensor.
    It mirrors the properties of the input tensor unless specified otherwise.

    Args:
        input (Tensor): The input tensor whose shape determines the output tensor's size.
        dtype (torch.dtype, optional): The desired data type of the returned tensor. Default is None.
        layout (torch.layout, optional): The desired layout of the returned tensor. Default is None.
        device (torch.device, optional): The desired device of the returned tensor. Default is None.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default is False.
        memory_format (torch.memory_format, optional): The desired memory format of the returned tensor. Default is torch.preserve_format.

    Returns:
        Tensor: A tensor of the same size as the input, filled with ones.
    """
    return torch.ones(input.size())

##################################################################################################################################################


import torch

def test_ones_like():
    results = {}

    # Test case 1: Basic test with default parameters
    input_tensor = torch.randn(2, 3, device='cuda')
    results["test_case_1"] = ones_like(input_tensor)

    # Test case 2: Test with a different dtype
    input_tensor = torch.randn(2, 3, device='cuda')
    results["test_case_2"] = ones_like(input_tensor, dtype=torch.float64)

    # Test case 3: Test with requires_grad=True
    input_tensor = torch.randn(2, 3, device='cuda')
    results["test_case_3"] = ones_like(input_tensor, requires_grad=True)

    # Test case 4: Test with a different device
    input_tensor = torch.randn(2, 3, device='cuda')
    results["test_case_4"] = ones_like(input_tensor, device='cuda')

    return results

test_results = test_ones_like()
