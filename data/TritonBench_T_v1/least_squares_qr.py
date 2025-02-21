import torch

def least_squares_qr(A: torch.Tensor, b: torch.Tensor, *, mode: str='reduced', out: torch.Tensor=None) -> torch.Tensor:
    """
    Solves the least squares problem for an overdetermined system of linear equations using QR decomposition.
    
    Args:
        A (Tensor): Coefficient matrix of shape (*, m, n), where * is zero or more batch dimensions.
        b (Tensor): Right-hand side vector or matrix of shape (*, m) or (*, m, k), where k is the number of right-hand sides.
        mode (str, optional): Determines the type of QR decomposition to use. One of 'reduced' (default) or 'complete'.
        out (Tensor, optional): Output tensor. Ignored if None. Default: None.

    Returns:
        Tensor: Least squares solution x.
    """
    Q, R = torch.linalg.qr(A, mode=mode)
    QTb = torch.matmul(Q.transpose(-2, -1).conj(), b)
    x = torch.linalg.solve(R, QTb)
    if out is not None:
        out.copy_(x)
        return out
    return x


##################################################################################################################################################


import torch

def test_least_squares_qr():
    results = {}
    
    # Test case 1: Simple overdetermined system with reduced QR
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')
    b1 = torch.tensor([7.0, 8.0, 9.0], device='cuda')
    results["test_case_1"] = least_squares_qr(A1, b1)
    
    # Test case 4: Multiple right-hand sides with reduced QR
    A4 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')
    b4 = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], device='cuda')
    results["test_case_4"] = least_squares_qr(A4, b4)
    
    return results

# Run the test
test_results = test_least_squares_qr()
