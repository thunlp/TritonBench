import torch


def std(input: torch.Tensor, dim=None, correction=1, keepdim=False, out=None) -> torch.Tensor:
    """
    Calculates the standard deviation over the specified dimensions of the input tensor.

    Parameters:
        input (torch.Tensor): The input tensor.
        dim (int or tuple of ints, optional): The dimension or dimensions to reduce.
        correction (int, optional): The correction factor for degrees of freedom. Defaults to 1 (Bessel's correction).
        keepdim (bool, optional): Whether to retain reduced dimensions with size 1. Defaults to False.
        out (torch.Tensor, optional): The output tensor.

    Returns:
        torch.Tensor: The standard deviation tensor.
    """
    return torch.std(input, dim=dim, keepdim=keepdim)

##################################################################################################################################################


import torch

def test_std():
    results = {}

    # Test case 1: Basic test with default parameters
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
    results["test_case_1"] = std(input_tensor)

    # Test case 2: Test with dim parameter
    input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')
    results["test_case_2"] = std(input_tensor, dim=0)

    # Test case 3: Test with keepdim=True
    input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')
    results["test_case_3"] = std(input_tensor, dim=1, keepdim=True)

    # Test case 4: Test with correction=0 (population standard deviation)
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
    results["test_case_4"] = std(input_tensor, correction=0)

    return results

test_results = test_std()
