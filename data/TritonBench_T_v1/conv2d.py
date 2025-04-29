import torch
import torch.nn.functional as F

def conv2d(
        input: torch.Tensor, 
        weight: torch.Tensor, 
        bias: torch.Tensor=None, 
        stride: int=1, 
        padding: int=0, 
        dilation: int=1, 
        groups: int=1) -> torch.Tensor:
    """
    Applies a 2D convolution over an input image composed of several input planes.

    Args:
        input (torch.Tensor): Input tensor of shape (minibatch, in_channels, iH, iW).
        weight (torch.Tensor): Filters (kernels) tensor of shape (out_channels, in_channels/groups, kH, kW).
        bias (torch.Tensor, optional): Bias tensor of shape (out_channels). Default: None.
        stride (int or tuple, optional): The stride of the convolution. Default: 1.
        padding (int or tuple, optional): Padding for input tensor. Default: 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.
        groups (int, optional): Split input into groups. Default: 1.

    Returns:
        torch.Tensor: Output tensor after applying the convolution.
    """
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

##################################################################################################################################################


import torch

def test_conv2d():
    results = {}

    # Test case 1: Basic convolution with default parameters
    input1 = torch.randn(1, 3, 5, 5, device='cuda')
    weight1 = torch.randn(2, 3, 3, 3, device='cuda')
    results["test_case_1"] = conv2d(input1, weight1)

    # Test case 2: Convolution with stride
    input2 = torch.randn(1, 3, 5, 5, device='cuda')
    weight2 = torch.randn(2, 3, 3, 3, device='cuda')
    results["test_case_2"] = conv2d(input2, weight2, stride=2)

    # Test case 3: Convolution with padding
    input3 = torch.randn(1, 3, 5, 5, device='cuda')
    weight3 = torch.randn(2, 3, 3, 3, device='cuda')
    results["test_case_3"] = conv2d(input3, weight3, padding=1)

    # Test case 4: Convolution with dilation
    input4 = torch.randn(1, 3, 5, 5, device='cuda')
    weight4 = torch.randn(2, 3, 3, 3, device='cuda')
    results["test_case_4"] = conv2d(input4, weight4, dilation=2)

    return results

test_results = test_conv2d()
print(test_results)