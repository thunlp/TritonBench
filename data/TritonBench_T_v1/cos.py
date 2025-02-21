import torch

def cos(input_tensor):
    """
    计算输入张量中每个元素的余弦值并返回一个新的张量。
    
    参数:
    input_tensor (torch.Tensor): 输入的张量。
    
    返回:
    torch.Tensor: 计算每个元素的余弦值后的新张量。
    """
    return torch.cos(input_tensor)

##################################################################################################################################################


import torch

def test_cos():
    results = {}

    # Test case 1: Single positive value
    input_tensor_1 = torch.tensor([0.0], device='cuda')
    results["test_case_1"] = cos(input_tensor_1)

    # Test case 2: Single negative value
    input_tensor_2 = torch.tensor([-3.14159265 / 2], device='cuda')
    results["test_case_2"] = cos(input_tensor_2)

    # Test case 3: Multiple values
    input_tensor_3 = torch.tensor([0.0, 3.14159265 / 2, 3.14159265], device='cuda')
    results["test_case_3"] = cos(input_tensor_3)

    # Test case 4: Large tensor
    input_tensor_4 = torch.linspace(-3.14159265, 3.14159265, steps=1000, device='cuda')
    results["test_case_4"] = cos(input_tensor_4)

    return results

test_results = test_cos()
