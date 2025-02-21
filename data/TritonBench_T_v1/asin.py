import torch

def asin(input_tensor):
    """
    计算输入张量的反正弦（arcsine）值。

    参数：
        input_tensor (torch.Tensor): 输入的张量。
        
    返回：
        torch.Tensor: 输入张量每个元素的反正弦值。如果输入值超出[-1, 1]范围，返回NaN。
    """
    if not isinstance(input_tensor, torch.Tensor):
        raise ValueError('The input must be a torch.Tensor.')
    return torch.asin(input_tensor)

##################################################################################################################################################


import torch

def test_asin():
    results = {}

    # Test case 1: Valid input within range [-1, 1]
    input_tensor_1 = torch.tensor([0.0, 0.5, -0.5, 1.0, -1.0], device='cuda')
    results["test_case_1"] = asin(input_tensor_1)

    # Test case 2: Input values exceeding the range [-1, 1]
    input_tensor_2 = torch.tensor([1.5, -1.5], device='cuda')
    results["test_case_2"] = asin(input_tensor_2)

    # Test case 3: Empty tensor
    input_tensor_3 = torch.tensor([], device='cuda')
    results["test_case_3"] = asin(input_tensor_3)

    # Test case 4: Single element tensor
    input_tensor_4 = torch.tensor([0.707], device='cuda')
    results["test_case_4"] = asin(input_tensor_4)

    return results

test_results = test_asin()
