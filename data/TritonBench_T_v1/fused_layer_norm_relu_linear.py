import torch
import torch.nn.functional as F


def fused_layer_norm_relu_linear(
        input: torch.Tensor, 
        weight: torch.Tensor, 
        bias: torch.Tensor=None, 
        normalized_shape: torch.Size=None, 
        eps: float=1e-05) -> torch.Tensor:
    """
    Applies a fused operation consisting of a linear transformation followed by ReLU activation 
    and layer normalization on the input tensor.

    Args:
        input (torch.Tensor): Input tensor with shape (*, in_features).
        weight (torch.Tensor): Weights for the linear transformation, shape (out_features, in_features).
        bias (torch.Tensor, optional): Bias for linear transformation, shape (out_features). Default is None.
        normalized_shape (int or list or torch.Size, optional): Shape of the dimensions to normalize.
        eps (float, optional): A value added to the denominator for numerical stability. Default is 1e-5.

    Returns:
        torch.Tensor: Result after applying the linear transformation, ReLU, and layer normalization.

    Example:
        >>> input = torch.randn(4, 5)  # Example input tensor
        >>> weight = torch.randn(3, 5)  # Linear transformation weights
        >>> bias = torch.randn(3)  # Bias for linear layer
        >>> normalized_shape = 3
        >>> output = fused_layer_norm_relu_linear(input, weight, bias, normalized_shape)
        >>> print(output.shape)  # Expected output shape: (4, 3)
    """
    linear_output = F.linear(input, weight, bias)

    relu_output = F.relu(linear_output)

    # Ensure normalized_shape is always passed as a tuple
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)

    normalized_output = F.layer_norm(relu_output, normalized_shape, eps=eps)

    return normalized_output

##################################################################################################################################################


import torch

def test_fused_layer_norm_relu_linear():
    results = {}

    # Test case 1: Basic test with bias
    input1 = torch.randn(4, 5, device='cuda')
    weight1 = torch.randn(3, 5, device='cuda')
    bias1 = torch.randn(3, device='cuda')
    normalized_shape1 = 3
    results["test_case_1"] = fused_layer_norm_relu_linear(input1, weight1, bias1, normalized_shape1)

    # Test case 2: Without bias
    input2 = torch.randn(4, 5, device='cuda')
    weight2 = torch.randn(3, 5, device='cuda')
    normalized_shape2 = 3
    results["test_case_2"] = fused_layer_norm_relu_linear(input2, weight2, None, normalized_shape2)

    # Test case 3: Different normalized shape
    input3 = torch.randn(4, 5, device='cuda')
    weight3 = torch.randn(3, 5, device='cuda')
    bias3 = torch.randn(3, device='cuda')
    normalized_shape3 = torch.Size([3])
    results["test_case_3"] = fused_layer_norm_relu_linear(input3, weight3, bias3, normalized_shape3)

    # Test case 4: Different epsilon value
    input4 = torch.randn(4, 5, device='cuda')
    weight4 = torch.randn(3, 5, device='cuda')
    bias4 = torch.randn(3, device='cuda')
    normalized_shape4 = 3
    eps4 = 1e-3
    results["test_case_4"] = fused_layer_norm_relu_linear(input4, weight4, bias4, normalized_shape4, eps=eps4)

    return results

test_results = test_fused_layer_norm_relu_linear()
print(test_results)