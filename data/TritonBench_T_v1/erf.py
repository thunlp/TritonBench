import torch

def erf(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Computes the error function (erf) of the input tensor element-wise.

    The error function is a special function of sigmoid shape that occurs in probability, 
    statistics, and partial differential equations. It is defined as:
    
    erf(x) = (2/√π) ∫[0 to x] exp(-t²) dt

    Args:
        input_tensor (Tensor): Input tensor to compute error function values for.

    Returns:
        Tensor: A tensor of the same shape as input containing the error function 
               values. Output values are in the range [-1, 1].

    Examples:
        >>> x = torch.tensor([0.0, 0.5, -0.5])
        >>> erf(x)
        tensor([0.0000, 0.5205, -0.5205])
    """
    return torch.special.erf(input_tensor)

##################################################################################################################################################


import torch
torch.manual_seed(42)

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
print(test_results)