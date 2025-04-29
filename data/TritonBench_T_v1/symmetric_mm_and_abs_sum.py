import torch

def symmetric_mm_and_abs_sum(
        A: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """
    Args:
        A (Tensor): Input matrix of shape `(n, m)` for which the symmetric product with its transpose is calculated.
        C (Tensor): Matrix of the same shape as `alpha * torch.mm(A, A.T)` to accumulate the scaled result.
        alpha (float): Scaling factor for the matrix product.
        beta (float): Scaling factor for matrix `C`.

    Returns:
        Tensor: Scalar tensor representing the sum of absolute values of the resulting matrix `C`.
    """
    C = alpha * torch.mm(A, A.T) + beta * C
    asum = torch.sum(torch.abs(C))
    return asum

##################################################################################################################################################


import torch

def test_symmetric_mm_and_abs_sum():
    results = {}

    # Test case 1: Basic test with small matrices
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    C1 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    alpha1 = 1.0
    beta1 = 1.0
    results["test_case_1"] = symmetric_mm_and_abs_sum(A1, C1, alpha1, beta1).item()

    # Test case 2: Test with different alpha and beta
    A2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    C2 = torch.tensor([[1.0, 1.0], [1.0, 1.0]], device='cuda')
    alpha2 = 0.5
    beta2 = 2.0
    results["test_case_2"] = symmetric_mm_and_abs_sum(A2, C2, alpha2, beta2).item()

    # Test case 3: Test with zero matrix for A
    A3 = torch.zeros((2, 2), device='cuda')
    C3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    alpha3 = 1.0
    beta3 = 1.0
    results["test_case_3"] = symmetric_mm_and_abs_sum(A3, C3, alpha3, beta3).item()

    # Test case 4: Test with negative values in A and C
    A4 = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]], device='cuda')
    C4 = torch.tensor([[-0.5, -0.5], [-0.5, -0.5]], device='cuda')
    alpha4 = 1.0
    beta4 = 1.0
    results["test_case_4"] = symmetric_mm_and_abs_sum(A4, C4, alpha4, beta4).item()

    return results

test_results = test_symmetric_mm_and_abs_sum()
print(test_results)