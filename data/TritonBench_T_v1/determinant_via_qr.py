import torch

def determinant_via_qr(A, *, mode='reduced', out=None):
    """
    Computes the determinant of a square matrix using QR decomposition.
    
    Parameters:
        A (Tensor): The input square matrix (n x n).
        mode (str, optional): The mode for QR decomposition ('reduced' or 'complete'). Defaults to 'reduced'.
        out (Tensor, optional): The output tensor to store the result. Defaults to None.
    
    Returns:
        Tensor: The determinant of the matrix A.
    """
    (Q, R) = torch.linalg.qr(A, mode=mode)
    det_Q = torch.det(Q)
    diag_R = torch.diagonal(R, dim1=-2, dim2=-1)
    prod_diag_R = torch.prod(diag_R, dim=-1)
    determinant = det_Q * prod_diag_R
    if out is not None:
        out.copy_(determinant)
        return out
    return determinant

##################################################################################################################################################


import torch

def test_determinant_via_qr():
    results = {}

    # Test case 1: 2x2 matrix, reduced mode
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_1"] = determinant_via_qr(A1)

    # Test case 2: 3x3 matrix, reduced mode
    A2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], device='cuda')
    results["test_case_2"] = determinant_via_qr(A2)

    # Test case 3: 2x2 matrix, complete mode
    A3 = torch.tensor([[2.0, 3.0], [1.0, 4.0]], device='cuda')
    results["test_case_3"] = determinant_via_qr(A3, mode='complete')

    # Test case 4: 3x3 matrix, complete mode
    A4 = torch.tensor([[2.0, 0.0, 1.0], [1.0, 3.0, 2.0], [4.0, 1.0, 3.0]], device='cuda')
    results["test_case_4"] = determinant_via_qr(A4, mode='complete')

    return results

test_results = test_determinant_via_qr()
