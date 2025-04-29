import torch
from typing import Optional

def pseudoinverse_svd(
        A: torch.Tensor, 
        full_matrices: bool=True, 
        rcond: float=1e-15, 
        out: Optional[torch.Tensor]=None) -> torch.Tensor:
    """
    Computes the pseudoinverse of a matrix using its singular value decomposition.

    Args:
        A (torch.Tensor): The input matrix.
        full_matrices (bool, optional): Whether to compute full matrices. Default is True.
        rcond (float, optional): The condition number cutoff for singular values. Default is 1e-15.
        out (torch.Tensor, optional): The output tensor.

    Returns:
        torch.Tensor: The pseudoinverse of the input matrix.
    """
    U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices)
    # Invert singular values larger than rcond * max(S)
    cutoff = rcond * S.max(dim=-1, keepdim=True).values
    S_inv = torch.where(S > cutoff, 1 / S, torch.zeros_like(S))
    # Create diagonal matrix of inverted singular values
    S_inv_mat = torch.diag_embed(S_inv)
    # Compute pseudoinverse
    A_pinv = Vh.transpose(-2, -1).conj() @ S_inv_mat @ U.transpose(-2, -1).conj()
    if out is not None:
        out.copy_(A_pinv)
        return out
    return A_pinv

##################################################################################################################################################


import torch

def test_pseudoinverse_svd():
    results = {}

    # Test case 1: Square matrix
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_1"] = pseudoinverse_svd(A1)

    # Test case 4: Singular matrix
    A4 = torch.tensor([[1.0, 2.0], [2.0, 4.0]], device='cuda')
    results["test_case_4"] = pseudoinverse_svd(A4)

    return results

test_results = test_pseudoinverse_svd()
print(test_results)