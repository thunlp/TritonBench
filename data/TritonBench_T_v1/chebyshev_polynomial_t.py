import torch


def chebyshev_polynomial_t(
    input: torch.Tensor, 
    n: int, 
    out: torch.Tensor=None) -> torch.Tensor:
    """
    Computes the Chebyshev polynomial of the first kind T_n(input).

    Args:
        input (torch.Tensor): The input tensor.
        n (int): Degree of the polynomial.
        out (torch.Tensor, optional): The output tensor.

    Returns:
        torch.Tensor: The Chebyshev polynomial of degree n evaluated at `input`.
    """
    return torch.special.chebyshev_polynomial_t(input, n)

##################################################################################################################################################


import torch

def test_chebyshev_polynomial_t():
    results = {}

    # Test case 1: Basic test with n=0
    input_tensor_1 = torch.tensor([0.5, -0.5, 0.0], device='cuda')
    n_1 = 0
    results["test_case_1"] = chebyshev_polynomial_t(input_tensor_1, n_1)

    # Test case 2: Basic test with n=1
    input_tensor_2 = torch.tensor([0.5, -0.5, 0.0], device='cuda')
    n_2 = 1
    results["test_case_2"] = chebyshev_polynomial_t(input_tensor_2, n_2)

    # Test case 3: Higher degree polynomial n=3
    input_tensor_3 = torch.tensor([0.5, -0.5, 0.0], device='cuda')
    n_3 = 3
    results["test_case_3"] = chebyshev_polynomial_t(input_tensor_3, n_3)

    # Test case 4: Negative input values with n=2
    input_tensor_4 = torch.tensor([-1.0, -0.5, -0.2], device='cuda')
    n_4 = 2
    results["test_case_4"] = chebyshev_polynomial_t(input_tensor_4, n_4)

    return results

test_results = test_chebyshev_polynomial_t()
print(test_results)