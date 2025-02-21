import torch
import torch.nn as nn
import torch.nn.functional as F

def dropout_relu_batch_norm_conv2d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor=None, stride: int=1, padding: int=0, dilation: int=1, groups: int=1, p: float=0.5, training: bool=True, inplace: bool=False) -> torch.Tensor:
    """
    Applies a 2D convolution followed by batch normalization, ReLU activation, and dropout.
    Sequentially applies conv2d, batch normalization for stabilizing training and reducing internal covariate shift,
    ReLU activation function, and dropout where some elements of the tensor are randomly zeroed with probability `p`.
    
    Args:
        input (Tensor): Input tensor of shape (N, C_in, H, W).
        weight (Tensor): Convolution filters of shape (C_out, C_in / groups, kH, kW).
        bias (Tensor, optional): Bias tensor of shape (C_out). Default is None.
        stride (int or tuple, optional): Stride of the convolution. Default is 1.
        padding (int, tuple, or str, optional): Implicit padding on both sides of the input. Default is 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Default is 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default is 1.
        p (float, optional): Probability of an element to be zeroed in dropout. Default is 0.5.
        training (bool, optional): If True, applies dropout during training. Default is True.
        inplace (bool, optional): If True, performs the operation in-place. Default is False.
    
    Returns:
        Tensor: The output tensor after applying conv2d, batch normalization, ReLU, and dropout.
    """
    conv_output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    bn_output = F.batch_norm(conv_output, running_mean=None, running_var=None, weight=None, bias=None, training=training)
    relu_output = F.relu(bn_output, inplace=inplace)
    output = F.dropout(relu_output, p=p, training=training, inplace=inplace)
    return output

##################################################################################################################################################


def test_dropout_relu_batch_norm_conv2d():
    # Initialize test results dictionary
    test_results = {}

    # Test case 1: Basic test with default parameters
    input_tensor = torch.randn(1, 3, 8, 8, device='cuda')
    weight_tensor = torch.randn(6, 3, 3, 3, device='cuda')
    bias_tensor = torch.randn(6, device='cuda')
    test_results["test_case_1"] = dropout_relu_batch_norm_conv2d(input_tensor, weight_tensor, bias_tensor)

    # Test case 2: Test with stride and padding
    test_results["test_case_2"] = dropout_relu_batch_norm_conv2d(input_tensor, weight_tensor, bias_tensor, stride=2, padding=1)

    # Test case 3: Test with different dropout probability
    test_results["test_case_3"] = dropout_relu_batch_norm_conv2d(input_tensor, weight_tensor, bias_tensor, p=0.3)

    # Test case 4: Test with groups
    weight_tensor_groups = torch.randn(6, 1, 3, 3, device='cuda')  # Adjust weight shape for groups
    input_tensor_groups = torch.randn(1, 6, 8, 8, device='cuda')   # Adjust input shape for groups
    test_results["test_case_4"] = dropout_relu_batch_norm_conv2d(input_tensor_groups, weight_tensor_groups, bias_tensor, groups=6)

    return test_results

# Execute the test function
test_results = test_dropout_relu_batch_norm_conv2d()
