import torch

def signbit(input: torch.Tensor, out: torch.Tensor=None) -> torch.Tensor:
    """
    Tests if each element of the input tensor has its sign bit set or not.
    This handles signed zeros, so negative zero (-0) returns True.

    Args:
    - input (torch.Tensor): The input tensor.
    - out (torch.Tensor, optional): The output tensor (default is None).

    Returns:
    - torch.Tensor: A tensor with the same shape as `input`, with boolean values indicating the sign bit status.
    """
    return torch.signbit(input, out=out)

##################################################################################################################################################


import torch

def test_signbit():
    results = {}

    # Test case 1: Positive and negative values
    input_tensor_1 = torch.tensor([1.0, -1.0, 0.0, -0.0], device='cuda')
    results["test_case_1"] = signbit(input_tensor_1)

    # Test case 2: All positive values
    input_tensor_2 = torch.tensor([3.5, 2.2, 0.1], device='cuda')
    results["test_case_2"] = signbit(input_tensor_2)

    # Test case 3: All negative values
    input_tensor_3 = torch.tensor([-3.5, -2.2, -0.1], device='cuda')
    results["test_case_3"] = signbit(input_tensor_3)

    # Test case 4: Mixed values with large numbers
    input_tensor_4 = torch.tensor([1e10, -1e10, 1e-10, -1e-10], device='cuda')
    results["test_case_4"] = signbit(input_tensor_4)

    return results

test_results = test_signbit()
