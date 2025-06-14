import torch
import torch.nn.functional as F

def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05):
    """
    Applies Batch Normalization to each channel across a batch of data.
    
    Parameters:
        input (Tensor): Input tensor with shape (N, C, H, W) for 4D input (e.g., images).
        running_mean (Tensor): Running mean for each channel, used in evaluation mode.
        running_var (Tensor): Running variance for each channel, used in evaluation mode.
        weight (Tensor, optional): Learnable scaling parameter for each channel.
        bias (Tensor, optional): Learnable bias for each channel.
        training (bool): Whether to use the statistics from the current batch or the running statistics.
        momentum (float): The value used to update running_mean and running_var.
        eps (float): A small value added to the denominator for numerical stability.

    Returns:
        Tensor: The normalized output.
    """
    return F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)

##################################################################################################################################################


import torch
import torch.nn.functional as F

# def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05):
#     """
#     Applies Batch Normalization to each channel across a batch of data.
    
#     Parameters:
#         input (Tensor): Input tensor with shape (N, C, H, W) for 4D input (e.g., images).
#         running_mean (Tensor): Running mean for each channel, used in evaluation mode.
#         running_var (Tensor): Running variance for each channel, used in evaluation mode.
#         weight (Tensor, optional): Learnable scaling parameter for each channel.
#         bias (Tensor, optional): Learnable bias for each channel.
#         training (bool): Whether to use the statistics from the current batch or the running statistics.
#         momentum (float): The value used to update running_mean and running_var.
#         eps (float): A small value added to the denominator for numerical stability.

#     Returns:
#         Tensor: The normalized output.
#     """
#     return F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)

def test_batch_norm():
    results = {}

    # Test case 1: Basic test with training=False
    input = torch.randn(2, 3, 4, 4, device='cuda')
    running_mean = torch.zeros(3, device='cuda')
    running_var = torch.ones(3, device='cuda')
    results["test_case_1"] = batch_norm(input, running_mean, running_var)

    # Test case 2: Test with training=True
    input = torch.randn(2, 3, 4, 4, device='cuda')
    running_mean = torch.zeros(3, device='cuda')
    running_var = torch.ones(3, device='cuda')
    results["test_case_2"] = batch_norm(input, running_mean, running_var, training=True)

    # Test case 3: Test with weight and bias
    input = torch.randn(2, 3, 4, 4, device='cuda')
    running_mean = torch.zeros(3, device='cuda')
    running_var = torch.ones(3, device='cuda')
    weight = torch.randn(3, device='cuda')
    bias = torch.randn(3, device='cuda')
    results["test_case_3"] = batch_norm(input, running_mean, running_var, weight, bias)

    # Test case 4: Test with different momentum and eps
    input = torch.randn(2, 3, 4, 4, device='cuda')
    running_mean = torch.zeros(3, device='cuda')
    running_var = torch.ones(3, device='cuda')
    results["test_case_4"] = batch_norm(input, running_mean, running_var, momentum=0.2, eps=1e-03)

    return results

test_results = test_batch_norm()
