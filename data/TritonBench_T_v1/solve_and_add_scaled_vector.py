import torch


def solve_and_add_scaled_vector(
        A: torch.Tensor, 
        b: torch.Tensor, 
        y: torch.Tensor, 
        alpha: float) -> torch.Tensor:
    """
    Solves a triangular system and adds a scaled vector.

    Args:
        A (torch.Tensor): The upper triangular matrix.
        b (torch.Tensor): The right-hand side vector.
        y (torch.Tensor): The vector to be scaled and added.
        alpha (float): The scaling factor.
    
    Returns:
        torch.Tensor: The solution vector.
    """
    x = torch.linalg.solve_triangular(A, b, upper=True)
    x += alpha * y
    return x

##################################################################################################################################################


import torch

def test_solve_and_add_scaled_vector():
    results = {}

    # Test case 1: Basic test with 2x2 upper triangular matrix
    A1 = torch.tensor([[2.0, 1.0], [0.0, 3.0]], device='cuda')
    b1 = torch.tensor([[5.0, 6.0], [7.0, 8]], device='cuda')
    y1 = torch.tensor([1.0, 2.0], device='cuda')
    alpha1 = 0.5
    results["test_case_1"] = solve_and_add_scaled_vector(A1, b1, y1, alpha1)
    return results

test_results = test_solve_and_add_scaled_vector()
print(test_results)