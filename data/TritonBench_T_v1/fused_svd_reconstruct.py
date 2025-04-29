import torch

def fused_svd_reconstruct(A: torch.Tensor) -> torch.Tensor:
    """
    Reconstructs the input matrix `A` using its Singular Value Decomposition (SVD).
    
    Parameters:
    A (torch.Tensor): The input matrix of shape (m, n).

    Returns:
    torch.Tensor: The reconstructed matrix A_reconstructed, approximating the original matrix A.
    """
    (U, S, Vh) = torch.linalg.svd(A, full_matrices=False)
    A_reconstructed = U @ torch.diag(S) @ Vh
    return A_reconstructed

##################################################################################################################################################


import torch
torch.manual_seed(42)

def test_fused_svd_reconstruct():
    results = {}
    
    # Test case 1: Square matrix
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_1"] = fused_svd_reconstruct(A1)
    
    # Test case 2: Rectangular matrix (more rows than columns)
    A2 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')
    results["test_case_2"] = fused_svd_reconstruct(A2)
    
    # Test case 3: Rectangular matrix (more columns than rows)
    A3 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')
    results["test_case_3"] = fused_svd_reconstruct(A3)
    
    # Test case 4: Single value matrix
    A4 = torch.tensor([[42.0]], device='cuda')
    results["test_case_4"] = fused_svd_reconstruct(A4)
    
    return results

test_results = test_fused_svd_reconstruct()
print(test_results)