import torch
from typing import Optional

def matrix_multiply_and_row_dot(
        A: torch.Tensor, 
        B: torch.Tensor, 
        alpha: float, 
        beta: float, 
        C: torch.Tensor) -> torch.Tensor:
    """
    Perform a scaled matrix-matrix multiplication and then calculate the dot product
    of the first two rows of the resulting matrix.

    Args:
    A (torch.Tensor): First input matrix of shape (n, m).
    B (torch.Tensor): Second input matrix of shape (m, p).
    alpha (float): Scalar multiplier for the matrix-matrix product.
    beta (float): Scalar multiplier for the input matrix `C`.
    C (torch.Tensor): Output matrix of shape (n, p) where the results are added.

    Returns:
    torch.Tensor: The dot product of the first two rows of the updated matrix C.
    """
    C = alpha * torch.mm(A, B) + beta * C
    result = torch.dot(C[0], C[1])
    return result

##################################################################################################################################################


import torch
torch.manual_seed(42)

def test_matrix_multiply_and_row_dot():
    results = {}

    # Test case 1
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    alpha = 1.0
    beta = 0.0
    C = torch.tensor([[0.0, 0.0], [0.0, 0.0]], device='cuda')
    results["test_case_1"] = matrix_multiply_and_row_dot(A, B, alpha, beta, C).item()

    # Test case 2
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    alpha = 0.5
    beta = 0.5
    C = torch.tensor([[1.0, 1.0], [1.0, 1.0]], device='cuda')
    results["test_case_2"] = matrix_multiply_and_row_dot(A, B, alpha, beta, C).item()

    # Test case 3
    A = torch.tensor([[2.0, 3.0], [4.0, 5.0]], device='cuda')
    B = torch.tensor([[6.0, 7.0], [8.0, 9.0]], device='cuda')
    alpha = 1.0
    beta = 1.0
    C = torch.tensor([[1.0, 1.0], [1.0, 1.0]], device='cuda')
    results["test_case_3"] = matrix_multiply_and_row_dot(A, B, alpha, beta, C).item()

    # Test case 4
    A = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    B = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    alpha = 2.0
    beta = 0.5
    C = torch.tensor([[2.0, 2.0], [2.0, 2.0]], device='cuda')
    results["test_case_4"] = matrix_multiply_and_row_dot(A, B, alpha, beta, C).item()

    return results

test_results = test_matrix_multiply_and_row_dot()
print(test_results)