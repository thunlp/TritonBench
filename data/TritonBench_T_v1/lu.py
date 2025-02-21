import torch

def lu(A, pivot=True, out=None):
    """
    Computes the LU decomposition of a matrix (or batch of matrices) using torch.linalg.lu.
    
    Args:
        A (Tensor): Input tensor of shape `(*, m, n)`, where `*` represents zero or more batch dimensions.
        pivot (bool, optional): If True, performs LU decomposition with partial pivoting. Default is True.
        out (tuple, optional): Tuple of three tensors to store the output. Defaults to None.
        
    Returns:
        tuple: (P, L, U) where:
            - P is the permutation matrix (only if pivot=True),
            - L is the lower triangular matrix (with ones on the diagonal),
            - U is the upper triangular matrix.
            
    Example:
        >>> A = torch.randn(3, 2)
        >>> P, L, U = compute_lu_decomposition(A)
        >>> P
        tensor([[0., 1., 0.],
                [0., 0., 1.],
                [1., 0., 0.]])
        >>> L
        tensor([[1.0000, 0.0000],
                [0.5007, 1.0000],
                [0.0633, 0.9755]])
        >>> U
        tensor([[0.3771, 0.0489],
                [0.0000, 0.9644]])
        >>> torch.dist(A, P @ L @ U)
        tensor(5.9605e-08)

        >>> A = torch.randn(2, 5, 7, device="cuda")
        >>> P, L, U = compute_lu_decomposition(A, pivot=False)
        >>> P
        tensor([], device='cuda:0')
        >>> torch.dist(A, L @ U)
        tensor(1.0376e-06, device='cuda:0')
    """
    (P, L, U) = torch.linalg.lu(A, pivot=pivot, out=out)
    return (P, L, U)

##################################################################################################################################################


import torch

def test_lu():
    results = {}

    # Test case 1: 2x2 matrix with pivoting
    A1 = torch.randn(2, 2, device="cuda")
    P1, L1, U1 = lu(A1)
    results["test_case_1"] = (P1.cpu(), L1.cpu(), U1.cpu())

    # Test case 2: 3x3 matrix with pivoting
    A2 = torch.randn(3, 3, device="cuda")
    P2, L2, U2 = lu(A2)
    results["test_case_2"] = (P2.cpu(), L2.cpu(), U2.cpu())

    # Test case 3: 2x3 matrix without pivoting
    A3 = torch.randn(2, 3, device="cuda")
    P3, L3, U3 = lu(A3, pivot=False)
    results["test_case_3"] = (P3.cpu(), L3.cpu(), U3.cpu())

    # Test case 4: Batch of 2x2 matrices with pivoting
    A4 = torch.randn(4, 2, 2, device="cuda")
    P4, L4, U4 = lu(A4)
    results["test_case_4"] = (P4.cpu(), L4.cpu(), U4.cpu())

    return results

test_results = test_lu()
