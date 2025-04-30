import torch
import torch.nn.functional as F
from typing import Optional

def sigmoid_conv2d(
        input: torch.Tensor, 
        weight: torch.Tensor, 
        bias: Optional[torch.Tensor]=None, 
        stride: int=1, 
        padding: int=0, 
        dilation: int=1,
        groups: int=1, 
        out: Optional[torch.Tensor]=None) -> torch.Tensor:
    """
    Performs a 2D convolution followed by a sigmoid activation.

    Args:
        input (torch.Tensor): Input tensor of shape (minibatch, in_channels, iH, iW).
        weight (torch.Tensor): Convolution filters of shape (out_channels, in_channels / groups, kH, kW).
        bias (torch.Tensor, optional): Bias tensor of shape (out_channels). Default: None.
        stride (int or tuple, optional): The stride of the convolution kernel. Default: 1.
        padding (int, tuple, or string, optional): Padding on both sides of the input. Default: 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.
        groups (int, optional): Number of groups to split the input into. Default: 1.
        out (torch.Tensor, optional): Output tensor.

    Returns:
        Tensor: The result of applying convolution followed by the sigmoid activation function.
    """
    conv_result = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    result = torch.sigmoid(conv_result)
    if out is not None:
        out.copy_(result)
    return result

##################################################################################################################################################


import torch
torch.manual_seed(42)
def test_sigmoid_conv2d():
    results = {}

    

    # Test case 1: Basic test with no bias, stride, padding, dilation, or groups
    input1 = torch.randn(1, 3, 5, 5, device='cuda')
    weight1 = torch.randn(2, 3, 3, 3, device='cuda')
    results["test_case_1"] = sigmoid_conv2d(input1, weight1)

    # Test case 2: Test with bias
    bias2 = torch.randn(2, device='cuda')
    results["test_case_2"] = sigmoid_conv2d(input1, weight1, bias=bias2)

    # Test case 3: Test with stride
    results["test_case_3"] = sigmoid_conv2d(input1, weight1, stride=2)

    # Test case 4: Test with padding
    results["test_case_4"] = sigmoid_conv2d(input1, weight1, padding=1)

    return results

test_results = test_sigmoid_conv2d()
print(test_results)