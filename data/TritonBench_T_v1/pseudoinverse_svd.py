import torch

def pseudoinverse_svd(A, full_matrices=True, rcond=1e-15, out=None):
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
