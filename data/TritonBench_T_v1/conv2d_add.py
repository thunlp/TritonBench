import torch
import torch.nn.functional as F


def conv2d_add(
        input: torch.Tensor, 
        weight: torch.Tensor, 
        bias: torch.Tensor = None, 
        other: torch.Tensor = None, 
        stride: int = 1, 
        padding: int = 0, 
        dilation: int = 1, 
        groups: int = 1, 
        alpha: float = 1, 
        out: torch.Tensor = None) -> torch.Tensor:
    """
    Applies a 2D convolution over an input image using specified filters and an optional bias, 
    then adds another tensor or scalar to the convolution result, scaled by alpha.
    
    Parameters:
        input (Tensor): The input tensor of shape (minibatch, in_channels, iH, iW).
        weight (Tensor): The convolution filters of shape (out_channels, in_channels / groups, kH, kW).
        bias (Tensor, optional): Optional bias tensor of shape (out_channels). Default: None.
        other (Tensor or Number, optional): The tensor or number to add to the convolution result. Default: None.
        stride (int or tuple, optional): The stride of the convolution kernel. Can be a single number or a tuple (sH, sW). Default: 1.
        padding (int, tuple, or string, optional): Padding on both sides of the input. Can be 'valid', 'same', single number, or tuple (padH, padW). Default: 0.
        dilation (int or tuple, optional): The spacing between kernel elements. Default: 1.
        groups (int, optional): Number of groups to split the input into, must divide in_channels and out_channels. Default: 1.
        alpha (Number, optional): The multiplier for other. Default: 1.
        out (Tensor, optional): The output tensor. Default: None.
    
    Returns:
        Tensor: The result of the convolution operation with the added value (scaled by alpha).
    """
    result = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    if other is not None:
        result = result + alpha * other
    return result

##################################################################################################################################################


def test_conv2d_add():
    results = {}

    # Test case 1: Basic convolution with bias, no addition
    input_tensor = torch.randn(1, 3, 5, 5, device='cuda')
    weight_tensor = torch.randn(2, 3, 3, 3, device='cuda')
    bias_tensor = torch.randn(2, device='cuda')
    results["test_case_1"] = conv2d_add(input_tensor, weight_tensor, bias=bias_tensor)

    # Test case 2: Convolution with addition of a scalar
    input_tensor = torch.randn(1, 3, 5, 5, device='cuda')
    weight_tensor = torch.randn(2, 3, 3, 3, device='cuda')
    scalar_addition = 2.0
    results["test_case_2"] = conv2d_add(input_tensor, weight_tensor, other=scalar_addition)

    # Test case 3: Convolution with addition of a tensor
    input_tensor = torch.randn(1, 3, 5, 5, device='cuda')
    weight_tensor = torch.randn(2, 3, 3, 3, device='cuda')
    other_tensor = torch.randn(1, 2, 3, 3, device='cuda')
    results["test_case_3"] = conv2d_add(input_tensor, weight_tensor, other=other_tensor)

    # Test case 4: Convolution with addition of a tensor and alpha scaling
    input_tensor = torch.randn(1, 3, 5, 5, device='cuda')
    weight_tensor = torch.randn(2, 3, 3, 3, device='cuda')
    other_tensor = torch.randn(1, 2, 3, 3, device='cuda')
    alpha_value = 0.5
    results["test_case_4"] = conv2d_add(input_tensor, weight_tensor, other=other_tensor, alpha=alpha_value)

    return results

test_results = test_conv2d_add()
print(test_results)