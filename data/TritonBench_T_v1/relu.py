import torch
import torch.nn.functional as F

def relu(input: torch.Tensor, inplace: bool=False) -> torch.Tensor:
    return F.relu(input, inplace=inplace)

##################################################################################################################################################


import torch
import torch.nn.functional as F

# def relu(input: torch.Tensor, inplace: bool=False) -> torch.Tensor:
#     return F.relu(input, inplace=inplace)

def test_relu():
    results = {}
    
    # Test case 1: Basic test with a simple tensor
    input1 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_1"] = relu(input1)
    
    # Test case 2: Test with a 2D tensor
    input2 = torch.tensor([[-1.0, 2.0], [3.0, -4.0]], device='cuda')
    results["test_case_2"] = relu(input2)
    
    # Test case 3: Test with inplace=True
    input3 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_3"] = relu(input3, inplace=True)
    
    # Test case 4: Test with a larger tensor
    input4 = torch.tensor([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], device='cuda')
    results["test_case_4"] = relu(input4)
    
    return results

test_results = test_relu()
