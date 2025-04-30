import torch

def cholesky_solve(
        B: torch.Tensor, 
        L: torch.Tensor, 
        upper: bool = False, 
        out: torch.Tensor = None) -> torch.Tensor:
    """
    Computes the solution to the linear system of equations for a symmetric positive-definite matrix given its Cholesky decomposition.
    
    Args:
        B (Tensor): The right-hand side tensor, with shape (*, n, k), where * represents zero or more batch dimensions.
        L (Tensor): A tensor of shape (*, n, n) representing the Cholesky decomposition of a symmetric or Hermitian positive-definite matrix, containing either the lower or upper triangle.
        upper (bool, optional): Flag indicating whether L is upper triangular. Defaults to False (meaning L is lower triangular).
        out (Tensor, optional): The output tensor. If None, a new tensor is returned.
    
    Returns:
        Tensor: The solution matrix X, with the same shape as B.
    """
    return torch.cholesky_solve(B, L, upper=upper, out=out)

##################################################################################################################################################


import torch
torch.manual_seed(42)

def test_cholesky_solve():
    results = {}

    # Test case 1: Lower triangular matrix
    B1 = torch.tensor([[1.0], [2.0]], device='cuda')
    L1 = torch.tensor([[2.0, 0.0], [1.0, 1.0]], device='cuda')
    results["test_case_1"] = cholesky_solve(B1, L1)

    # Test case 2: Upper triangular matrix
    B2 = torch.tensor([[1.0], [2.0]], device='cuda')
    L2 = torch.tensor([[2.0, 1.0], [0.0, 1.0]], device='cuda')
    results["test_case_2"] = cholesky_solve(B2, L2, upper=True)

    # Test case 3: Batch of matrices, lower triangular
    B3 = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]], device='cuda')
    L3 = torch.tensor([[[2.0, 0.0], [1.0, 1.0]], [[3.0, 0.0], [1.0, 2.0]]], device='cuda')
    results["test_case_3"] = cholesky_solve(B3, L3)

    # Test case 4: Batch of matrices, upper triangular
    B4 = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]], device='cuda')
    L4 = torch.tensor([[[2.0, 1.0], [0.0, 1.0]], [[3.0, 1.0], [0.0, 2.0]]], device='cuda')
    results["test_case_4"] = cholesky_solve(B4, L4, upper=True)

    return results

test_results = test_cholesky_solve()
print(test_results)