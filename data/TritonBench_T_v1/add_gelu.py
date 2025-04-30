import torch
import torch.nn.functional as F
import torch

def add_gelu(
        input: torch.Tensor, 
        other: torch.Tensor, 
        alpha: float = 1, 
        approximate: str = 'none', 
        out: torch.Tensor = None) -> torch.Tensor:
    """
    Adds the tensor or number `other`, scaled by the multiplier `alpha`, to the input tensor `input`,
    and then applies the Gaussian Error Linear Units (GELU) activation function to the result.
    
    Args:
        input (Tensor): The input tensor.
        other (Tensor or Number): The tensor or number to add to input.
        alpha (Number, optional): The multiplier for other. Default is 1.
        approximate (str, optional): The approximation method for GELU. Default is 'none'.
        out (Tensor, optional): The output tensor.

    Returns:
        Tensor: The result of the operation.
    """
    result = input + alpha * other
    if approximate == 'none':
        result = F.gelu(result)
    elif approximate == 'tanh':
        result = 0.5 * result * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (result + 0.044715 * result ** 3)))
    else:
        raise ValueError("Invalid value for 'approximate'. Expected 'none' or 'tanh'.")
    return result

##################################################################################################################################################


import torch
torch.manual_seed(42)

def test_add_gelu():
    results = {}

    # Test case 1: Basic test with default parameters
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other_tensor = torch.tensor([0.5, 0.5, 0.5], device='cuda')
    results["test_case_1"] = add_gelu(input_tensor, other_tensor)

    # Test case 2: Test with alpha parameter
    alpha = 2
    results["test_case_2"] = add_gelu(input_tensor, other_tensor, alpha=alpha)

    # Test case 3: Test with approximate='tanh'
    approximate = 'tanh'
    results["test_case_3"] = add_gelu(input_tensor, other_tensor, approximate=approximate)

    # Test case 4: Test with a scalar 'other'
    other_scalar = 0.5
    results["test_case_4"] = add_gelu(input_tensor, other_scalar)

    return results

test_results = test_add_gelu()
print(test_results)