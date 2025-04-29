import torch

def gammaln(input: torch.Tensor, out: torch.Tensor=None) -> torch.Tensor:
    """
    Computes the natural logarithm of the absolute value of the gamma function on the input tensor.
    
    Args:
        input (torch.Tensor): the input tensor.
        out (torch.Tensor, optional): the output tensor.

    Returns:
        torch.Tensor: tensor containing the natural log of the gamma function for each element in the input.
    """
    return torch.special.gammaln(input, out=out)

##################################################################################################################################################


import torch

def test_gammaln():
    results = {}
    
    # Test case 1: Single value tensor
    input1 = torch.tensor([2.0], device='cuda')
    results["test_case_1"] = gammaln(input1)
    
    # Test case 2: Multi-value tensor
    input2 = torch.tensor([2.0, 3.0, 4.0], device='cuda')
    results["test_case_2"] = gammaln(input2)
    
    # Test case 3: Tensor with negative values
    input3 = torch.tensor([-2.5, -3.5, -4.5], device='cuda')
    results["test_case_3"] = gammaln(input3)
    
    # Test case 4: Large tensor
    input4 = torch.tensor([i for i in range(1, 1001)], dtype=torch.float32, device='cuda')
    results["test_case_4"] = gammaln(input4)
    
    return results

test_results = test_gammaln()
print(test_results)