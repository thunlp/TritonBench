import torch

def erf(input_tensor):
    """
    计算输入张量的误差函数（error function）。

    参数：
    input_tensor (Tensor): 输入的张量。

    返回：
    Tensor: 输入张量中每个元素的误差函数值。
    """
    return torch.special.erf(input_tensor)

##################################################################################################################################################


import torch

def erf(input_tensor):
    """
    计算输入张量的误差函数（error function）。

    参数：
    input_tensor (Tensor): 输入的张量。

    返回：
    Tensor: 输入张量中每个元素的误差函数值。
    """
    return torch.special.erf(input_tensor)

def test_erf():
    results = {}
    
    # Test case 1: Single element tensor
    input_tensor = torch.tensor([0.5], device='cuda')
    results["test_case_1"] = erf(input_tensor)
    
    # Test case 2: Multi-element tensor
    input_tensor = torch.tensor([0.5, -1.0, 2.0], device='cuda')
    results["test_case_2"] = erf(input_tensor)
    
    # Test case 3: Large values tensor
    input_tensor = torch.tensor([10.0, -10.0], device='cuda')
    results["test_case_3"] = erf(input_tensor)
    
    # Test case 4: Zero tensor
    input_tensor = torch.tensor([0.0], device='cuda')
    results["test_case_4"] = erf(input_tensor)
    
    return results

test_results = test_erf()
