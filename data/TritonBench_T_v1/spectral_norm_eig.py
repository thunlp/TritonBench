import torch

def spectral_norm_eig(A, *, out=None):
    """
    Computes the spectral norm (operator norm induced by the Euclidean vector norm)
    of a square matrix using its eigenvalues.

    Args:
        A (Tensor): Tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                    consisting of square matrices.
        out (Tensor, optional): Output tensor. Ignored if `None`. Default: `None`.

    Returns:
        Tensor: The spectral norm of the input matrix or batch of matrices.
    """
    (eigenvalues, _) = torch.linalg.eig(A)
    abs_eigenvalues = torch.abs(eigenvalues)
    (spectral_norm, _) = torch.max(abs_eigenvalues, dim=-1)
    if out is not None:
        out.copy_(spectral_norm)
        return out
    return spectral_norm

##################################################################################################################################################


import torch

def test_spectral_norm_eig():
    results = {}

    # Test case 1: Single 2x2 matrix
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_1"] = spectral_norm_eig(A1)

    # Test case 2: Batch of 2x2 matrices
    A2 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device='cuda')
    results["test_case_2"] = spectral_norm_eig(A2)

    # Test case 3: Single 3x3 matrix
    A3 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], device='cuda')
    results["test_case_3"] = spectral_norm_eig(A3)

    # Test case 4: Batch of 3x3 matrices
    A4 = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], 
                       [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]], device='cuda')
    results["test_case_4"] = spectral_norm_eig(A4)

    return results

test_results = test_spectral_norm_eig()
