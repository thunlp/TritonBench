import torch


def tanh_linear(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor=None) -> torch.Tensor:
    """
    Applies a linear transformation followed by a Tanh activation.

    Args:
        input (torch.Tensor): The input tensor of shape (*, in_features).
        weight (torch.Tensor): The weight matrix of shape (out_features, in_features).
        bias (torch.Tensor, optional): The optional bias tensor of shape (out_features). Default: None.

    Returns:
        torch.Tensor: The result of applying the linear transformation followed by Tanh activation.

    Example:
        >>> import torch
        >>> from tanh_linear import tanh_linear
        >>> input = torch.randn(5, 3)  # (batch_size, in_features)
        >>> weight = torch.randn(4, 3)  # (out_features, in_features)
        >>> bias = torch.randn(4)       # (out_features)
        >>> result = tanh_linear(input, weight, bias)
        >>> result.shape
        torch.Size([5, 4])
    """
    output = torch.matmul(input, weight.t())
    if bias is not None:
        output += bias
    return torch.tanh(output)

##################################################################################################################################################


import torch
torch.manual_seed(42)
def test_tanh_linear():
    results = {}

    # Test case 1: input, weight, and bias on GPU
    input1 = torch.randn(5, 3, device='cuda')
    weight1 = torch.randn(4, 3, device='cuda')
    bias1 = torch.randn(4, device='cuda')
    result1 = tanh_linear(input1, weight1, bias1)
    results["test_case_1"] = result1

    # Test case 2: input and weight on GPU, bias is None
    input2 = torch.randn(5, 3, device='cuda')
    weight2 = torch.randn(4, 3, device='cuda')
    result2 = tanh_linear(input2, weight2)
    results["test_case_2"] = result2

    # Test case 3: input and weight on GPU, bias on GPU
    input3 = torch.randn(2, 3, device='cuda')
    weight3 = torch.randn(2, 3, device='cuda')
    bias3 = torch.randn(2, device='cuda')
    result3 = tanh_linear(input3, weight3, bias3)
    results["test_case_3"] = result3

    # Test case 4: input, weight, and bias on GPU with different dimensions
    input4 = torch.randn(3, 2, device='cuda')
    weight4 = torch.randn(2, 2, device='cuda')
    bias4 = torch.randn(2, device='cuda')
    result4 = tanh_linear(input4, weight4, bias4)
    results["test_case_4"] = result4

    return results

test_results = test_tanh_linear()
print(test_results)