import torch
import torch.nn.functional as F
from typing import Optional, Union, Tuple
def fused_instance_norm_selu_conv2d(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int], str] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        eps: float = 1e-05,
        momentum: float = 0.1) -> torch.Tensor:
    """
    Performs a fused operation consisting of Conv2d, followed by SELU activation,
    and finally Instance Normalization.

    Args:
        input (torch.Tensor): Input tensor of shape (N, C_in, H, W).
        weight (torch.Tensor): Convolution filters of shape (C_out, C_in/groups, kH, kW).
        bias (Optional[torch.Tensor]): Optional bias tensor for conv2d of shape (C_out). Default: None.
        stride (Union[int, Tuple[int, int]]): Stride for the convolution. Default: 1.
        padding (Union[int, Tuple[int, int], str]): Padding for the convolution. Default: 0.
        dilation (Union[int, Tuple[int, int]]): Dilation for the convolution. Default: 1.
        groups (int): Number of groups for the convolution. Default: 1.
        eps (float): A value added to the denominator for numerical stability in InstanceNorm. Default: 1e-05.
        momentum (float): The value used for the running_mean and running_var computation in InstanceNorm.
                          Has effect only when running stats are provided (not used in this basic F.instance_norm call). Default: 0.1.

    Returns:
        torch.Tensor: The result of applying Conv2d -> SELU -> InstanceNorm to the input tensor.
    """
    conv_output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    selu_output = F.selu(conv_output)
    normalized_output = F.instance_norm(selu_output, eps=eps, momentum=momentum)
    return normalized_output

##################################################################################################################################################


import torch
torch.manual_seed(42)

def test_fused_instance_norm_selu_conv2d():
    results = {}
    
    # Test case 1: Basic test with default parameters
    input_tensor = torch.randn(1, 3, 5, 5, device='cuda')
    weight_tensor = torch.randn(3, 3, 3, 3, device='cuda')
    results["test_case_1"] = fused_instance_norm_selu_conv2d(input_tensor, weight_tensor)
    
    # Test case 2: Test with stride
    results["test_case_2"] = fused_instance_norm_selu_conv2d(input_tensor, weight_tensor, stride=2)
    
    # Test case 3: Test with padding
    results["test_case_3"] = fused_instance_norm_selu_conv2d(input_tensor, weight_tensor, padding=1)
    
    return results

test_results = test_fused_instance_norm_selu_conv2d()
print(test_results)