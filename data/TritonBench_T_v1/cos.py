import torch

def cos(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Computes the cosine of each element in the input tensor and returns a new tensor.
    
    Args:
        input_tensor (torch.Tensor): The input tensor.
    
    Returns:
        torch.Tensor: A new tensor containing the cosine values of each element.
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
print(test_results)