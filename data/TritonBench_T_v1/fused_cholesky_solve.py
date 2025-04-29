import torch

def fused_cholesky_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Solve the equation Ax = b using the Cholesky decomposition of the symmetric positive-definite matrix A.

    Args:
        A (torch.Tensor): The symmetric positive-definite matrix A of shape (n, n).
        b (torch.Tensor): The right-hand side tensor b of shape (n, k).

    Returns:
        torch.Tensor: The solution tensor x of shape (n, k).
    """
    L = torch.cholesky(A)
    y = torch.linalg.solve(L, b)
    x = torch.linalg.solve(L.T, y)
    return x

##################################################################################################################################################


import torch
torch.manual_seed(42)

def test_fused_cholesky_solve():
    results = {}

    # Test case 1: Simple 2x2 positive-definite matrix
    A1 = torch.tensor([[4.0, 1.0], [1.0, 3.0]], device='cuda')
    b1 = torch.tensor([[1.0], [2.0]], device='cuda')
    results["test_case_1"] = fused_cholesky_solve(A1, b1)

    # Test case 2: Larger 3x3 positive-definite matrix
    A2 = torch.tensor([[6.0, 2.0, 1.0], [2.0, 5.0, 2.0], [1.0, 2.0, 4.0]], device='cuda')
    b2 = torch.tensor([[1.0], [2.0], [3.0]], device='cuda')
    results["test_case_2"] = fused_cholesky_solve(A2, b2)

    # Test case 3: 2x2 matrix with multiple right-hand sides
    A3 = torch.tensor([[5.0, 2.0], [2.0, 3.0]], device='cuda')
    b3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_3"] = fused_cholesky_solve(A3, b3)

    # Test case 4: 3x3 matrix with multiple right-hand sides
    A4 = torch.tensor([[7.0, 3.0, 1.0], [3.0, 6.0, 2.0], [1.0, 2.0, 5.0]], device='cuda')
    b4 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')
    results["test_case_4"] = fused_cholesky_solve(A4, b4)

    return results

test_results = test_fused_cholesky_solve()
print(test_results)