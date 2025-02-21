import torch

def qr(A, mode='reduced', out=None):
    """
    Computes the QR decomposition of a matrix (or batch of matrices).
    
    Args:
        A (Tensor): Input tensor of shape (*, m, n) where * is zero or more batch dimensions.
        mode (str, optional): One of 'reduced', 'complete', or 'r'. 
                              Controls the shape of the returned tensors. Default is 'reduced'.
        out (tuple, optional): Output tuple of two tensors. Ignored if None. Default is None.
    
    Returns:
        tuple: A tuple containing two tensors (Q, R), where:
            - Q is an orthogonal matrix (real case) or unitary matrix (complex case).
            - R is an upper triangular matrix with real diagonal.
    """
    (Q, R) = torch.linalg.qr(A, mode=mode)
    return (Q, R)

##################################################################################################################################################


import torch

def test_qr():
    results = {}

    # Test case 1: reduced mode, 2x2 matrix
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    Q1, R1 = qr(A1, mode='reduced')
    results["test_case_1"] = (Q1.cpu(), R1.cpu())

    # Test case 2: complete mode, 3x2 matrix
    A2 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')
    Q2, R2 = qr(A2, mode='complete')
    results["test_case_2"] = (Q2.cpu(), R2.cpu())

    # Test case 3: r mode, 2x3 matrix
    A3 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')
    Q3, R3 = qr(A3, mode='r')
    results["test_case_3"] = (Q3.cpu(), R3.cpu())

    # Test case 4: reduced mode, batch of 2x2 matrices
    A4 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device='cuda')
    Q4, R4 = qr(A4, mode='reduced')
    results["test_case_4"] = (Q4.cpu(), R4.cpu())

    return results

test_results = test_qr()
