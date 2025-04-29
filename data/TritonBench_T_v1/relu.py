import torch

def relu(input: torch.Tensor, inplace: bool=False) -> torch.Tensor:
    """
    Applies the rectified linear unit (ReLU) function to each element in input.

    Args:
        input (torch.Tensor): The input tensor.
        inplace (bool, optional): If True, will perform the operation in-place. Default: False.

    Returns:
        torch.Tensor: The output tensor with ReLU applied.
    """
    max_val = torch.max(input, torch.zeros_like(input))
    if inplace:
        input.copy_(max_val)
        return input
    else:
        return max_val

##################################################################################################################################################


import torch

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
print(test_results)