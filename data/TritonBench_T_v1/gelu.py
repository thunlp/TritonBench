import torch
import torch.nn.functional as F

def gelu(input: torch.Tensor, approximate: str='none') -> torch.Tensor:
    return F.gelu(input, approximate=approximate)

##################################################################################################################################################


import torch
import torch.nn.functional as F

def gelu(input: torch.Tensor, approximate: str='none') -> torch.Tensor:
    return F.gelu(input, approximate=approximate)

def test_gelu():
    results = {}
    
    # Test case 1: Default approximate='none'
    input_tensor_1 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_1"] = gelu(input_tensor_1)
    
    # Test case 2: approximate='tanh'
    input_tensor_2 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_2"] = gelu(input_tensor_2, approximate='tanh')
    
    # Test case 3: Larger tensor with default approximate='none'
    input_tensor_3 = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device='cuda')
    results["test_case_3"] = gelu(input_tensor_3)
    
    # Test case 4: Larger tensor with approximate='tanh'
    input_tensor_4 = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device='cuda')
    results["test_case_4"] = gelu(input_tensor_4, approximate='tanh')
    
    return results

test_results = test_gelu()
