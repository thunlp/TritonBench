import torch

def determinant_lu(
        A: torch.Tensor, 
        *, 
        pivot: bool = True, 
        out: torch.Tensor = None) -> torch.Tensor:
    """
    Compute the determinant of a square matrix using LU decomposition.

    Args:
        A (Tensor): Tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions 
                    consisting of square matrices.
        pivot (bool, optional): Controls whether to compute the LU decomposition with partial 
                                 pivoting (True) or without pivoting (False). Default: True.
        out (Tensor, optional): Output tensor. Ignored if None. Default: None.

    Returns:
        Tensor: The determinant of the input matrix or batch of matrices.
    """
    (P, L, U) = torch.linalg.lu(A, pivot=pivot)
    diag_U = torch.diagonal(U, dim1=-2, dim2=-1)
    det_U = torch.prod(diag_U, dim=-1)
    if pivot:
        (sign_P, _) = torch.linalg.slogdet(P)
        det = sign_P * det_U
    else:
        det = det_U
    if out is not None:
        out.copy_(det)
        return out
    return det

##################################################################################################################################################


import torch
torch.manual_seed(42)

def test_determinant_lu():
    results = {}

    # Test case 1: 2x2 matrix with pivot=True
    A1 = torch.tensor([[3.0, 1.0], [2.0, 4.0]], device='cuda')
    results["test_case_1"] = determinant_lu(A1)

    # Test case 2: 3x3 matrix with pivot=False
    A2 = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]], device='cuda')
    results["test_case_2"] = determinant_lu(A2, pivot=False)

    # Test case 3: Batch of 2x2 matrices with pivot=True
    A3 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device='cuda')
    results["test_case_3"] = determinant_lu(A3)

    # Test case 4: 4x4 matrix with pivot=True
    A4 = torch.tensor([[1.0, 0.0, 2.0, -1.0],
                       [3.0, 0.0, 0.0, 5.0],
                       [2.0, 1.0, 4.0, -3.0],
                       [1.0, 0.0, 5.0, 0.0]], device='cuda')
    results["test_case_4"] = determinant_lu(A4)

    return results

test_results = test_determinant_lu()
print(test_results)