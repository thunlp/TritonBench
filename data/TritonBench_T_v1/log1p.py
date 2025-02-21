import torch

def log1p(input: torch.Tensor, out: torch.Tensor=None) -> torch.Tensor:
    """
    This function computes the natural logarithm of (1 + input).
    It is more accurate than torch.log for small values of input.

    Args:
    input (torch.Tensor): The input tensor.
    out (torch.Tensor, optional): The output tensor. Default is None.

    Returns:
    torch.Tensor: A tensor containing the natural logarithm of (1 + input).
    """
    return torch.log1p(input, out=out)

##################################################################################################################################################


import torch

def test_log1p():
    results = {}

    # Test case 1: Basic test with a small positive tensor
    input1 = torch.tensor([0.1, 0.2, 0.3], device='cuda')
    results["test_case_1"] = log1p(input1)

    # Test case 2: Test with a tensor containing zero
    input2 = torch.tensor([0.0, 0.5, 1.0], device='cuda')
    results["test_case_2"] = log1p(input2)

    # Test case 3: Test with a tensor containing negative values
    input3 = torch.tensor([-0.1, -0.2, -0.3], device='cuda')
    results["test_case_3"] = log1p(input3)

    # Test case 4: Test with a larger tensor
    input4 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
    results["test_case_4"] = log1p(input4)

    return results

test_results = test_log1p()
