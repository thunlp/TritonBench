import torch

def eig(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the eigenvalues and eigenvectors of a square matrix.

    Args:
        A (torch.Tensor): The input matrix.

    Returns:
        tuple: A tuple containing two tensors:
            - eigenvalues (torch.Tensor): The eigenvalues of the matrix.
            - eigenvectors (torch.Tensor): The eigenvectors of the matrix.
    """
    (eigenvalues, eigenvectors) = torch.linalg.eig(A)
    return (eigenvalues, eigenvectors)

##################################################################################################################################################


import torch
torch.manual_seed(42)

def test_eig():
    results = {}

    # Test case 1: 2x2 matrix with distinct eigenvalues
    A1 = torch.tensor([[2.0, 0.0], [0.0, 3.0]], device='cuda')
    results["test_case_1"] = eig(A1)

    # Test case 2: 2x2 matrix with repeated eigenvalues
    A2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    results["test_case_2"] = eig(A2)

    # Test case 3: 3x3 matrix with complex eigenvalues
    A3 = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device='cuda')
    results["test_case_3"] = eig(A3)

    # Test case 4: 3x3 matrix with real eigenvalues
    A4 = torch.tensor([[4.0, 1.0, 0.0], [1.0, 4.0, 0.0], [0.0, 0.0, 5.0]], device='cuda')
    results["test_case_4"] = eig(A4)

    return results

test_results = test_eig()
print(test_results)