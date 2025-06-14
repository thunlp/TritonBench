import torch

def digamma(input_tensor):
    """
    Computes the digamma function (logarithmic derivative of the gamma function) for the input tensor.

    Args:
    - input_tensor (torch.Tensor): The tensor on which to compute the digamma function.

    Returns:
    - torch.Tensor: A tensor containing the digamma values.
    """
    return torch.special.digamma(input_tensor)

##################################################################################################################################################


import torch

# def digamma(input_tensor):
#     """
#     Computes the digamma function (logarithmic derivative of the gamma function) for the input tensor.

#     Args:
#     - input_tensor (torch.Tensor): The tensor on which to compute the digamma function.

#     Returns:
#     - torch.Tensor: A tensor containing the digamma values.
#     """
#     return torch.special.digamma(input_tensor)

def test_digamma():
    results = {}
    
    # Test case 1: Single positive value
    input_tensor = torch.tensor([1.0], device='cuda')
    results["test_case_1"] = digamma(input_tensor)
    
    # Test case 2: Single negative value
    input_tensor = torch.tensor([-1.0], device='cuda')
    results["test_case_2"] = digamma(input_tensor)
    
    # Test case 3: Multiple positive values
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_3"] = digamma(input_tensor)
    
    # Test case 4: Mixed positive and negative values
    input_tensor = torch.tensor([1.0, -1.0, 2.0, -2.0], device='cuda')
    results["test_case_4"] = digamma(input_tensor)
    
    return results

test_results = test_digamma()
