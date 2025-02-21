import torch

def sqrt(input, out=None):
    """
    Computes the square root element-wise of the input tensor.
    
    Args:
        input (Tensor): The input tensor.
        out (Tensor, optional): The output tensor. Defaults to None.
    
    Returns:
        Tensor: A new tensor with the square root of the elements.
    """
    return torch.sqrt(input, out=out)

##################################################################################################################################################


import torch

def test_sqrt():
    results = {}

    # Test case 1: Simple positive numbers
    input1 = torch.tensor([4.0, 9.0, 16.0], device='cuda')
    results["test_case_1"] = sqrt(input1)

    # Test case 2: Including zero
    input2 = torch.tensor([0.0, 1.0, 4.0], device='cuda')
    results["test_case_2"] = sqrt(input2)

    # Test case 3: Large numbers
    input3 = torch.tensor([1e10, 1e20, 1e30], device='cuda')
    results["test_case_3"] = sqrt(input3)

    # Test case 4: Small numbers
    input4 = torch.tensor([1e-10, 1e-20, 1e-30], device='cuda')
    results["test_case_4"] = sqrt(input4)

    return results

test_results = test_sqrt()
