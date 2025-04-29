import torch

def matrix_multiply_symmetric(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """
    Perform matrix multiplication and symmetric matrix update.

    Args:
        A (Tensor): The first input matrix of shape `(n, m)`.
        B (Tensor): The second input matrix of shape `(m, p)`.
        C (Tensor): The target matrix for the operations, shape `(n, p)`.
        alpha (float): Scalar multiplier for matrix products.
        beta (float): Scalar multiplier for adding to `C`.

    Returns:
        Tensor: The updated matrix `C` after the operations.
    
    Example:
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        B = torch.tensor([[0.5, -1.0], [1.5, 2.0]])
        C = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        alpha, beta = 2.0, 0.5
        result = matrix_multiply_symmetric(A, B, C, alpha, beta)
        print(result)
    """
    C = alpha * torch.mm(A, B) + beta * C
    C = alpha * torch.mm(C, C.T) + beta * C
    return C

##################################################################################################################################################


import torch

def test_matrix_multiply_symmetric():
    results = {}

    # Test Case 1: Basic test with 2x2 matrices
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    B = torch.tensor([[0.5, -1.0], [1.5, 2.0]], device='cuda')
    C = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    alpha, beta = 2.0, 0.5
    results["test_case_1"] = matrix_multiply_symmetric(A, B, C, alpha, beta)

    # Test Case 2: Test with identity matrices
    A = torch.eye(3, device='cuda')
    B = torch.eye(3, device='cuda')
    C = torch.eye(3, device='cuda')
    alpha, beta = 1.0, 1.0
    results["test_case_2"] = matrix_multiply_symmetric(A, B, C, alpha, beta)

    # Test Case 3: Test with zero matrices
    A = torch.zeros((2, 2), device='cuda')
    B = torch.zeros((2, 2), device='cuda')
    C = torch.zeros((2, 2), device='cuda')
    alpha, beta = 1.0, 1.0
    results["test_case_3"] = matrix_multiply_symmetric(A, B, C, alpha, beta)

    # Test Case 4: Test with different alpha and beta
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    B = torch.tensor([[0.5, -1.0], [1.5, 2.0]], device='cuda')
    C = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    alpha, beta = 0.5, 2.0
    results["test_case_4"] = matrix_multiply_symmetric(A, B, C, alpha, beta)

    return results

test_results = test_matrix_multiply_symmetric()
print(test_results)