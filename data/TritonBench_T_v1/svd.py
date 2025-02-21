import torch

def svd(A, full_matrices=True):
    """
    Compute the Singular Value Decomposition (SVD) of a tensor.
    
    Args:
        A (Tensor): The input tensor of shape (*, m, n), where * represents zero or more batch dimensions.
        full_matrices (bool, optional): Whether to compute the full or reduced SVD. Default is True.
        
    Returns:
        tuple: A tuple (U, S, Vh) where:
            - U: Tensor of shape (*, m, m) or (*, m, k) depending on full_matrices.
            - S: Tensor of shape (*, k), where k is the number of singular values.
            - Vh: Tensor of shape (*, k, n) or (*, n, n) depending on full_matrices.
    """
    (U, S, Vh) = torch.linalg.svd(A, full_matrices=full_matrices)
    return (U, S, Vh)

##################################################################################################################################################


import torch

def test_svd():
    results = {}

    # Test case 1: 2x2 matrix, full_matrices=True
    A1 = torch.tensor([[3.0, 1.0], [1.0, 3.0]], device='cuda')
    U1, S1, Vh1 = svd(A1, full_matrices=True)
    results["test_case_1"] = (U1.cpu(), S1.cpu(), Vh1.cpu())

    # Test case 2: 3x2 matrix, full_matrices=False
    A2 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')
    U2, S2, Vh2 = svd(A2, full_matrices=False)
    results["test_case_2"] = (U2.cpu(), S2.cpu(), Vh2.cpu())

    # Test case 3: 2x3 matrix, full_matrices=True
    A3 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')
    U3, S3, Vh3 = svd(A3, full_matrices=True)
    results["test_case_3"] = (U3.cpu(), S3.cpu(), Vh3.cpu())

    # Test case 4: 3x3 matrix, full_matrices=False
    A4 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device='cuda')
    U4, S4, Vh4 = svd(A4, full_matrices=False)
    results["test_case_4"] = (U4.cpu(), S4.cpu(), Vh4.cpu())

    return results

test_results = test_svd()
