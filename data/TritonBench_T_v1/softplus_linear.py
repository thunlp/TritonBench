import torch
import torch.nn.functional as F


def softplus_linear(input, weight, bias=None, beta=1, threshold=20):
    """
    Applies a linear transformation to the input tensor, followed by the Softplus activation function.
    
    Args:
        input (Tensor): The input tensor of shape (batch_size, in_features).
        weight (Tensor): The weight tensor of shape (out_features, in_features).
        bias (Tensor, optional): The bias tensor of shape (out_features). Default: None.
        beta (float, optional): The scaling factor for the Softplus function. Default: 1.
        threshold (float, optional): The value above which the function becomes linear. Default: 20.
    
    Returns:
        Tensor: The output tensor after applying the linear transformation and Softplus activation.
    """
    linear_out = F.linear(input, weight, bias)
    softplus_out = 1 / beta * torch.log(1 + torch.exp(beta * linear_out))
    softplus_out = torch.where(linear_out > threshold, linear_out, softplus_out)
    return softplus_out

##################################################################################################################################################


import torch

def test_softplus_linear():
    results = {}

    # Test case 1: Basic test with default beta and threshold
    input1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight1 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    bias1 = torch.tensor([0.0, 0.0], device='cuda')
    results["test_case_1"] = softplus_linear(input1, weight1, bias1)

    # Test case 2: Test with non-default beta
    input2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight2 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    bias2 = torch.tensor([0.0, 0.0], device='cuda')
    results["test_case_2"] = softplus_linear(input2, weight2, bias2, beta=2)

    # Test case 3: Test with non-default threshold
    input3 = torch.tensor([[10.0, 20.0], [30.0, 40.0]], device='cuda')
    weight3 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    bias3 = torch.tensor([0.0, 0.0], device='cuda')
    results["test_case_3"] = softplus_linear(input3, weight3, bias3, threshold=15)

    # Test case 4: Test with no bias
    input4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight4 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    results["test_case_4"] = softplus_linear(input4, weight4)

    return results

test_results = test_softplus_linear()
