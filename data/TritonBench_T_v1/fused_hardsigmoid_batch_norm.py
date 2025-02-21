import torch
import torch.nn.functional as F
import torch

def fused_hardsigmoid_batch_norm(x: torch.Tensor, running_mean: torch.Tensor, running_var: torch.Tensor, weight: torch.Tensor=None, bias: torch.Tensor=None, training: bool=False, momentum: float=0.1, eps: float=1e-05, inplace: bool=False) -> torch.Tensor:
    normalized_x = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, training, momentum, eps)
    output = torch.nn.functional.hardsigmoid(normalized_x, inplace=inplace)
    return output

##################################################################################################################################################


import torch
import torch.nn.functional as F

def fused_hardsigmoid_batch_norm(x: torch.Tensor, running_mean: torch.Tensor, running_var: torch.Tensor, weight: torch.Tensor=None, bias: torch.Tensor=None, training: bool=False, momentum: float=0.1, eps: float=1e-05, inplace: bool=False) -> torch.Tensor:
    normalized_x = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, training, momentum, eps)
    output = torch.nn.functional.hardsigmoid(normalized_x, inplace=inplace)
    return output

def test_fused_hardsigmoid_batch_norm():
    results = {}
    
    # Test case 1: Default parameters
    x = torch.randn(10, 3, 32, 32, device='cuda')
    running_mean = torch.zeros(3, device='cuda')
    running_var = torch.ones(3, device='cuda')
    results["test_case_1"] = fused_hardsigmoid_batch_norm(x, running_mean, running_var)
    
    # Test case 2: With weight and bias
    weight = torch.randn(3, device='cuda')
    bias = torch.randn(3, device='cuda')
    results["test_case_2"] = fused_hardsigmoid_batch_norm(x, running_mean, running_var, weight, bias)
    
    # Test case 3: Training mode
    results["test_case_3"] = fused_hardsigmoid_batch_norm(x, running_mean, running_var, training=True)
    
    # Test case 4: Inplace operation
    results["test_case_4"] = fused_hardsigmoid_batch_norm(x, running_mean, running_var, inplace=True)
    
    return results

test_results = test_fused_hardsigmoid_batch_norm()
