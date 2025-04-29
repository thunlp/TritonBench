import torch

def det(A: torch.Tensor) -> torch.Tensor:
    """
    Computes the determinant of a square matrix.

    Args:
        A (torch.Tensor): The input matrix.

    Returns:
        torch.Tensor: The determinant of the input matrix.
    """
    return torch.linalg.det(A)

##################################################################################################################################################


import torch

def test_det():
    results = {}
    
    # Test case 1: 2x2 identity matrix
    A1 = torch.eye(2, device='cuda')
    results["test_case_1"] = det(A1).item()
    
    # Test case 2: 3x3 matrix with random values
    A2 = torch.rand((3, 3), device='cuda')
    results["test_case_2"] = det(A2).item()
    
    # Test case 3: 4x4 matrix with all zeros
    A3 = torch.zeros((4, 4), device='cuda')
    results["test_case_3"] = det(A3).item()
    
    # Test case 4: 2x2 matrix with specific values
    A4 = torch.tensor([[4.0, 7.0], [2.0, 6.0]], device='cuda')
    results["test_case_4"] = det(A4).item()
    
    return results

test_results = test_det()
print(test_results)