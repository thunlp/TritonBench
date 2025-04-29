import torch
import torch.nn.functional as F
from typing import Optional

def relu_max_pool2d_conv2d(
        input: torch.Tensor, 
        weight: torch.Tensor, 
        bias: Optional[torch.Tensor]=None, 
        conv_stride: int=1, 
        conv_padding: int=0, 
        conv_dilation: int=1, 
        conv_groups: int=1, 
        pool_kernel_size: int=2, 
        pool_stride: Optional[int]=None, 
        pool_padding: int=0, 
        pool_dilation: int=1, 
        pool_ceil_mode: bool=False, 
        inplace: bool=False) -> torch.Tensor:
    """
    Applies a 2D convolution followed by max pooling and then applies the ReLU activation function element-wise to the pooled result.

    Args:
        input (Tensor): The input tensor of shape (minibatch, in_channels, iH, iW).
        weight (Tensor): The convolution filters of shape (out_channels, in_channels / groups, kH, kW).
        bias (Tensor, optional): Optional bias tensor of shape (out_channels). Default: None.
        conv_stride (int or tuple, optional): The stride of the convolution kernel. Default: 1.
        conv_padding (int, tuple, or string, optional): Padding added to all sides of the input in convolution. Default: 0.
        conv_dilation (int or tuple, optional): The spacing between kernel elements in convolution. Default: 1.
        conv_groups (int, optional): Number of blocked connections from input channels to output channels in convolution. Default: 1.
        pool_kernel_size (int or tuple): The size of the pooling region in max pooling.
        pool_stride (int or tuple, optional): The stride of the pooling operation. Default: `pool_kernel_size`.
        pool_padding (int or tuple, optional): Padding added to all sides of the input in max pooling. Default: 0.
        pool_dilation (int or tuple, optional): The stride between elements within a sliding window in max pooling. Default: 1.
        pool_ceil_mode (bool, optional): If True, uses `ceil` instead of `floor` to compute output shape. Default: False.
        inplace (bool, optional): If True, performs ReLU in-place. Default: False.

    Returns:
        Tensor: The resulting tensor after the convolution, max pooling, and ReLU operations.
    """
    x = F.conv2d(input, weight, bias, stride=conv_stride, padding=conv_padding, dilation=conv_dilation, groups=conv_groups)
    x = F.max_pool2d(x, kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding, dilation=pool_dilation, ceil_mode=pool_ceil_mode)
    x = F.relu(x, inplace=inplace)
    return x

##################################################################################################################################################


import torch

def test_relu_max_pool2d_conv2d():
    results = {}

    torch.manual_seed(42)
    
    # Test case 1: Basic test with default parameters
    input = torch.randn(1, 3, 8, 8, device='cuda')
    weight = torch.randn(6, 3, 3, 3, device='cuda')
    results["test_case_1"] = relu_max_pool2d_conv2d(input, weight)
    
    # Test case 2: Test with bias
    bias = torch.randn(6, device='cuda')
    results["test_case_2"] = relu_max_pool2d_conv2d(input, weight, bias=bias)
    
    # Test case 3: Test with different convolution stride and padding
    results["test_case_3"] = relu_max_pool2d_conv2d(input, weight, conv_stride=2, conv_padding=1)
    
    # Test case 4: Test with different max pooling parameters
    results["test_case_4"] = relu_max_pool2d_conv2d(input, weight, pool_kernel_size=3, pool_stride=2, pool_padding=1)
    
    return results

test_results = test_relu_max_pool2d_conv2d()
print(test_results)