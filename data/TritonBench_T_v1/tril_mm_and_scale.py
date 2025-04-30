import torch

def tril_mm_and_scale(
        A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """
    Perform matrix multiplication of the lower triangular part of matrix A with matrix B,
    scales the result by alpha, and then scales the final output by beta.
    
    Parameters:
    A (torch.Tensor): A 2D matrix of shape (n, n).
    B (torch.Tensor): A matrix of shape (n, p) to be multiplied with the lower triangular part of A.
    alpha (float): Scaling factor for the matrix multiplication result.
    beta (float): Scaling factor for the final result.
    
    Returns:
    torch.Tensor: The final scaled matrix.
    """
    lower_tril_A = torch.tril(A)
    result = alpha * torch.mm(lower_tril_A, B)
    final_result = beta * result
    return final_result

##################################################################################################################################################


import torch
torch.manual_seed(42)
def test_tril_mm_and_scale():
    results = {}

    # Test case 1: Basic functionality with square matrices
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    B1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    alpha1 = 1.0
    beta1 = 1.0
    results["test_case_1"] = tril_mm_and_scale(A1, B1, alpha1, beta1)

    # Test case 2: Different alpha and beta values
    A2 = torch.tensor([[1.0, 0.0], [3.0, 4.0]], device='cuda')
    B2 = torch.tensor([[2.0, 3.0], [4.0, 5.0]], device='cuda')
    alpha2 = 0.5
    beta2 = 2.0
    results["test_case_2"] = tril_mm_and_scale(A2, B2, alpha2, beta2)

    # Test case 3: Larger matrix
    A3 = torch.tensor([[1.0, 0.0, 0.0], [4.0, 5.0, 0.0], [7.0, 8.0, 9.0]], device='cuda')
    B3 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')
    alpha3 = 1.0
    beta3 = 1.0
    results["test_case_3"] = tril_mm_and_scale(A3, B3, alpha3, beta3)

    # Test case 4: Zero matrix A
    A4 = torch.zeros((2, 2), device='cuda')
    B4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    alpha4 = 1.0
    beta4 = 1.0
    results["test_case_4"] = tril_mm_and_scale(A4, B4, alpha4, beta4)

    return results

test_results = test_tril_mm_and_scale()
print(test_results)