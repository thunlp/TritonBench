import torch

def fused_qr_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Solve the linear system Ax = b using QR decomposition.

    Args:
        A (torch.Tensor): The matrix A of shape (m, n) where m >= n.
        b (torch.Tensor): The right-hand side tensor b of shape (m, k).

    Returns:
        torch.Tensor: The solution tensor x of shape (n, k).
    """
    (Q, R) = torch.linalg.qr(A)
    Qt_b = torch.matmul(Q.T, b)
    x = torch.linalg.solve(R, Qt_b)
    return x

##################################################################################################################################################


import torch
torch.manual_seed(42)

def test_fused_qr_solve():
    results = {}

    # Test case 1: Square matrix A and vector b
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    b1 = torch.tensor([[5.0], [6.0]], device='cuda')
    results["test_case_1"] = fused_qr_solve(A1, b1)

    # Test case 2: Rectangular matrix A (m > n) and vector b
    A2 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')
    b2 = torch.tensor([[7.0], [8.0], [9.0]], device='cuda')
    results["test_case_2"] = fused_qr_solve(A2, b2)

    # Test case 3: Square matrix A and matrix b with multiple columns
    A3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    b3 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    results["test_case_3"] = fused_qr_solve(A3, b3)

    # Test case 4: Rectangular matrix A (m > n) and matrix b with multiple columns
    A4 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')
    b4 = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], device='cuda')
    results["test_case_4"] = fused_qr_solve(A4, b4)

    return results

test_results = test_fused_qr_solve()
print(test_results)