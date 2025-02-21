import torch
import torch.nn.functional as F

def fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups, eps=1e-05, *, out=None):
    """
    Fused operation combining element-wise addition, element-wise multiplication,
    and group normalization.

    Args:
        input1 (Tensor): The first input tensor X.
        input2 (Tensor): The second input tensor Y, must be broadcastable to the shape of X.
        weight (Tensor): Learnable weight parameter γ of shape (C,).
        bias (Tensor): Learnable bias parameter β of shape (C,).
        num_groups (int): Number of groups for group normalization.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-5.
        out (Tensor, optional): Output tensor. Ignored if None.

    Returns:
        Tensor: The output tensor after applying the fused operations.
    """
    z = torch.add(input1, input2)
    m = torch.mul(z, input2)
    o = torch.nn.functional.group_norm(m, num_groups=num_groups, weight=weight, bias=bias, eps=eps)
    if out is not None:
        out.copy_(o)
        return out
    return o

##################################################################################################################################################


import torch
import torch.nn.functional as F

def test_fused_add_mul_groupnorm():
    results = {}

    # Test case 1: Basic functionality test
    input1 = torch.randn(2, 4, 4, 4, device='cuda')
    input2 = torch.randn(2, 4, 4, 4, device='cuda')
    weight = torch.randn(4, device='cuda')
    bias = torch.randn(4, device='cuda')
    num_groups = 2
    results["test_case_1"] = fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups)

    # Test case 2: Different shapes for input1 and input2 (broadcastable)
    input1 = torch.randn(2, 4, 4, 4, device='cuda')
    input2 = torch.randn(1, 4, 1, 1, device='cuda')  # Broadcastable shape
    weight = torch.randn(4, device='cuda')
    bias = torch.randn(4, device='cuda')
    num_groups = 2
    results["test_case_2"] = fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups)

    # Test case 3: Single group normalization (equivalent to layer normalization)
    input1 = torch.randn(2, 4, 4, 4, device='cuda')
    input2 = torch.randn(2, 4, 4, 4, device='cuda')
    weight = torch.randn(4, device='cuda')
    bias = torch.randn(4, device='cuda')
    num_groups = 1
    results["test_case_3"] = fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups)

    # Test case 4: No weight and bias (should default to None)
    input1 = torch.randn(2, 4, 4, 4, device='cuda')
    input2 = torch.randn(2, 4, 4, 4, device='cuda')
    num_groups = 2
    results["test_case_4"] = fused_add_mul_groupnorm(input1, input2, None, None, num_groups)

    return results

test_results = test_fused_add_mul_groupnorm()
