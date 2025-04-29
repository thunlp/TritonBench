from typing import Optional
import torch

def invert_matrix_lu(
        A: torch.Tensor, 
        pivot: bool=True, 
        out: Optional[torch.Tensor]=None) -> torch.Tensor:
    """
    Computes the inverse of a square matrix using LU decomposition.

    Parameters:
        A (Tensor): A square invertible matrix.
        pivot (bool, optional): Whether to use partial pivoting (default: True).
        out (Tensor, optional): An output tensor to store the result (default: None).

    Returns:
        Tensor: The inverse of matrix A.
    """
    (P, L, U) = torch.linalg.lu(A, pivot=pivot)
    n = A.size(-1)
    if pivot:
        P_eye = torch.eye(n, device=A.device, dtype=A.dtype).expand_as(A)
        P_mat = P @ P_eye
    else:
        P_mat = torch.eye(n, device=A.device, dtype=A.dtype)
    Y = torch.linalg.solve(L, P_mat)
    A_inv = torch.linalg.solve(U, Y)
    if out is not None:
        out.copy_(A_inv)
        return out
    return A_inv

##################################################################################################################################################


import torch
torch.manual_seed(42)

def test_invert_matrix_lu():
    results = {}

    # Test case 1: Basic test with pivot=True
    A1 = torch.tensor([[4.0, 3.0], [6.0, 3.0]], device='cuda')
    results["test_case_1"] = invert_matrix_lu(A1)

    # Test case 2: Basic test with pivot=False
    A2 = torch.tensor([[4.0, 3.0], [6.0, 3.0]], device='cuda')
    results["test_case_2"] = invert_matrix_lu(A2, pivot=False)

    # Test case 3: Larger matrix with pivot=True
    A3 = torch.tensor([[7.0, 2.0, 1.0], [0.0, 3.0, -1.0], [-3.0, 4.0, 2.0]], device='cuda')
    results["test_case_3"] = invert_matrix_lu(A3)

    # Test case 4: Larger matrix with pivot=False
    A4 = torch.tensor([[7.0, 2.0, 1.0], [0.0, 3.0, -1.0], [-3.0, 4.0, 2.0]], device='cuda')
    results["test_case_4"] = invert_matrix_lu(A4, pivot=False)

    return results

test_results = test_invert_matrix_lu()
print(test_results)