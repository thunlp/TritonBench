import torch
import torch.nn.functional as F
from typing import Optional
def silu_batch_norm(
        input: torch.Tensor, 
        running_mean: torch.Tensor, 
        running_var: torch.Tensor, 
        weight: Optional[torch.Tensor]=None, 
        bias: Optional[torch.Tensor]=None, 
        training: bool=False, 
        momentum: float=0.1, eps: float=1e-05):
    """
    Applies Batch Normalization over an input tensor across channels, followed by
    the SiLU (Sigmoid Linear Unit) activation function element-wise.
    
    Args:
        input (Tensor): The input tensor for Batch Normalization.
        running_mean (Tensor): The running mean tensor (used during evaluation).
        running_var (Tensor): The running variance tensor (used during evaluation).
        weight (Tensor, optional): The weight tensor for Batch Normalization scaling. Default: None.
        bias (Tensor, optional): The bias tensor for Batch Normalization. Default: None.
        training (bool, optional): Whether the module is in training mode. Default: False.
        momentum (float, optional): Value used for the running mean and variance computation. Default: 0.1.
        eps (float, optional): A small value added to the denominator for numerical stability. Default: 1e-5.

    Returns:
        Tensor: The output tensor after applying batch normalization and SiLU activation.
    """
    bn_output = F.batch_norm(input, running_mean, running_var, weight=weight, bias=bias, training=training, momentum=momentum, eps=eps)
    output = bn_output * torch.sigmoid(bn_output)
    return output

##################################################################################################################################################


import torch
import torch.nn.functional as F

def test_silu_batch_norm():
    results = {}
    
    torch.manual_seed(42)
    # Test case 1: Basic functionality with training=False
    input_tensor = torch.randn(3, 5, device='cuda')
    running_mean = torch.zeros(5, device='cuda')
    running_var = torch.ones(5, device='cuda')
    results["test_case_1"] = silu_batch_norm(input_tensor, running_mean, running_var, training=False)

    # Test case 2: With weight and bias, training=False
    weight = torch.ones(5, device='cuda')
    bias = torch.zeros(5, device='cuda')
    results["test_case_2"] = silu_batch_norm(input_tensor, running_mean, running_var, weight=weight, bias=bias, training=False)

    # Test case 3: With training=True
    results["test_case_3"] = silu_batch_norm(input_tensor, running_mean, running_var, weight=weight, bias=bias, training=True)

    # Test case 4: Different momentum and eps values
    results["test_case_4"] = silu_batch_norm(input_tensor, running_mean, running_var, weight=weight, bias=bias, training=True, momentum=0.2, eps=1e-3)

    return results

test_results = test_silu_batch_norm()
print(test_results)