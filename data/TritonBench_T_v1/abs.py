import torch

def abs(input_tensor: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    """
    Computes the absolute value of each element in the input tensor.

    Args:
        input_tensor (Tensor): The input tensor.
        out (Tensor, optional): The output tensor to store the result. Default is None.

    Returns:
        Tensor: A tensor with the absolute values of the input tensor.
    """
    return torch.abs(input_tensor, out=out)

##################################################################################################################################################


import torch
torch.manual_seed(42)

def test_abs():
    results = {}

    # Test case 1: Simple positive and negative values
    input_tensor_1 = torch.tensor([-1.0, 2.0, -3.0], device='cuda')
    results["test_case_1"] = abs(input_tensor_1)

    # Test case 2: Zero values
    input_tensor_2 = torch.tensor([0.0, -0.0, 0.0], device='cuda')
    results["test_case_2"] = abs(input_tensor_2)

    # Test case 3: Mixed positive, negative, and zero values
    input_tensor_3 = torch.tensor([-5.0, 0.0, 5.0], device='cuda')
    results["test_case_3"] = abs(input_tensor_3)

    # Test case 4: Large positive and negative values
    input_tensor_4 = torch.tensor([-1e10, 1e10, -1e-10], device='cuda')
    results["test_case_4"] = abs(input_tensor_4)

    return results

test_results = test_abs()
print(test_results)