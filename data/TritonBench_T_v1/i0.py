import torch

def i0(input_tensor, out=None):
    """
    计算输入张量每个元素的零阶改良贝塞尔函数 (I_0)。

    Args:
        input_tensor (Tensor): 输入张量
        out (Tensor, optional): 输出张量，如果提供则保存结果
    
    Returns:
        Tensor: 每个元素应用 I_0 函数后的结果
    """
    return torch.special.i0(input_tensor, out=out)

##################################################################################################################################################


import torch

def test_i0():
    results = {}

    # Test case 1: Simple tensor on GPU
    input_tensor_1 = torch.tensor([0.0, 1.0, 2.0], device='cuda')
    results["test_case_1"] = i0(input_tensor_1)

    # Test case 2: Larger tensor with negative values on GPU
    input_tensor_2 = torch.tensor([-1.0, -2.0, 3.0, 4.0], device='cuda')
    results["test_case_2"] = i0(input_tensor_2)

    # Test case 3: Tensor with mixed positive and negative values on GPU
    input_tensor_3 = torch.tensor([-3.0, 0.0, 3.0], device='cuda')
    results["test_case_3"] = i0(input_tensor_3)

    # Test case 4: Tensor with fractional values on GPU
    input_tensor_4 = torch.tensor([0.5, 1.5, 2.5], device='cuda')
    results["test_case_4"] = i0(input_tensor_4)

    return results

test_results = test_i0()
