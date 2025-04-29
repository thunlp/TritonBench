import torch
import torch.nn.functional as F
from typing import Optional
def relu_conv2d(
        input: torch.Tensor, 
        weight: torch.Tensor, 
        bias: Optional[torch.Tensor]=None, 
        stride: int=1, 
        padding: int=0, 
        dilation: int=1, 
        groups: int=1, 
        inplace: bool=False) -> torch.Tensor:
    """
    Applies a 2D convolution followed by a ReLU activation function.

    Args:
        input (Tensor): The input tensor of shape (minibatch, in_channels, iH, iW).
        weight (Tensor): The convolution filters of shape (out_channels, in_channels / groups, kH, kW).
        bias (Tensor, optional): Optional bias tensor of shape (out_channels). Default: None.
        stride (int or tuple, optional): The stride of the convolution kernel. Default: 1.
        padding (int, tuple, or string, optional): Padding added to all sides of the input. Default: 0.
        dilation (int or tuple, optional): The spacing between kernel elements. Default: 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
        inplace (bool, optional): If True, will perform ReLU operation in-place. Default: False.

    Returns:
        Tensor: A tensor resulting from the 2D convolution followed by ReLU activation.
    """
    conv_out = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    return F.relu(conv_out, inplace=inplace)

##################################################################################################################################################


import torch
torch.manual_seed(42)

def test_relu_conv2d():
    results = {}
    
    # Test case 1: Basic test with default parameters
    input1 = torch.randn(1, 3, 5, 5, device='cuda')
    weight1 = torch.randn(2, 3, 3, 3, device='cuda')
    results["test_case_1"] = relu_conv2d(input1, weight1)
    
    # Test case 2: Test with bias
    input2 = torch.randn(1, 3, 5, 5, device='cuda')
    weight2 = torch.randn(2, 3, 3, 3, device='cuda')
    bias2 = torch.randn(2, device='cuda')
    results["test_case_2"] = relu_conv2d(input2, weight2, bias=bias2)
    
    # Test case 3: Test with stride and padding
    input3 = torch.randn(1, 3, 5, 5, device='cuda')
    weight3 = torch.randn(2, 3, 3, 3, device='cuda')
    results["test_case_3"] = relu_conv2d(input3, weight3, stride=2, padding=1)
    
    # Test case 4: Test with dilation and groups
    input4 = torch.randn(1, 4, 5, 5, device='cuda')
    weight4 = torch.randn(2, 2, 3, 3, device='cuda') # Corrected in_channels/groups for weight
    results["test_case_4"] = relu_conv2d(input4, weight4, dilation=2, groups=2)
    
    return results

test_results = test_relu_conv2d()
print(test_results)