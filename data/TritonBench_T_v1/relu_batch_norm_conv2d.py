import torch
import torch.nn.functional as F
from torch import nn

def relu_batch_norm_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, running_mean=None, running_var=None, bn_weight=None, bn_bias=None, training=False, momentum=0.1, eps=1e-05, inplace=False):
    """
    Applies a 2D convolution over the input tensor, followed by batch normalization 
    and then applies the ReLU activation function element-wise to the normalized result.
    
    Args:
        input (Tensor): The input tensor of shape (minibatch, in_channels, iH, iW).
        weight (Tensor): The convolution filters of shape (out_channels, in_channels / groups, kH, kW).
        bias (Tensor, optional): Optional bias tensor of shape (out_channels). Default: None.
        stride (int or tuple, optional): The stride of the convolution kernel. Default: 1.
        padding (int, tuple, or string, optional): Padding added to all sides of the input. Default: 0.
        dilation (int or tuple, optional): The spacing between kernel elements. Default: 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
        running_mean (Tensor, optional): The running mean for batch normalization. Default: None.
        running_var (Tensor, optional): The running variance for batch normalization. Default: None.
        bn_weight (Tensor, optional): Learnable scaling factor for batch normalization (gamma). Default: None.
        bn_bias (Tensor, optional): Learnable shift factor for batch normalization (beta). Default: None.
        training (bool, optional): If True, updates running statistics for batch normalization. Default: False.
        momentum (float, optional): Value for updating the running mean and variance in batch normalization. Default: 0.1.
        eps (float, optional): A small value added for numerical stability in batch normalization. Default: 1e-5.
        inplace (bool, optional): If True, performs ReLU in-place. Default: False.
    
    Returns:
        Tensor: The output tensor after convolution, batch normalization, and ReLU activation.
    """
    conv_result = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    bn_result = F.batch_norm(conv_result, running_mean, running_var, bn_weight, bn_bias, training=training, momentum=momentum, eps=eps)
    return F.relu(bn_result, inplace=inplace)


##################################################################################################################################################


import torch
import torch.nn.functional as F
from torch import nn

import torch
from torch import nn

# Define a simple test function
def test_relu_batch_norm_conv2d():
    # Define input tensor (batch_size, channels, height, width)
    input_tensor = torch.randn(4, 3, 32, 32)  # Example: 4 images, 3 channels, 32x32 resolution
    
    # Define convolution weight tensor (out_channels, in_channels/groups, kernel_height, kernel_width)
    weight_tensor = torch.randn(6, 3, 3, 3)  # Example: 6 filters, 3 input channels, 3x3 kernel
    
    # Define optional bias tensor (out_channels)
    bias_tensor = torch.randn(6)  # Example: bias for each of the 6 filters
    
    # Define batch normalization parameters
    running_mean = torch.zeros(6)
    running_var = torch.ones(6)
    bn_weight = torch.ones(6)
    bn_bias = torch.zeros(6)
    
    # Call the relu_batch_norm_conv2d function
    output_tensor = relu_batch_norm_conv2d(
        input=input_tensor,
        weight=weight_tensor,
        bias=bias_tensor,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        running_mean=running_mean,
        running_var=running_var,
        bn_weight=bn_weight,
        bn_bias=bn_bias,
        training=True,
        momentum=0.1,
        eps=1e-5,
        inplace=False
    )

    # Print the shape of the output tensor
    print(f"Output tensor shape: {output_tensor.shape}")
    
    # Check if output tensor has the expected shape
    expected_shape = (4, 6, 32, 32)  # 4 images, 6 output channels, 32x32 resolution
    assert output_tensor.shape == expected_shape, f"Expected shape {expected_shape}, but got {output_tensor.shape}"

    return output_tensor

# Run the test
output = test_relu_batch_norm_conv2d()