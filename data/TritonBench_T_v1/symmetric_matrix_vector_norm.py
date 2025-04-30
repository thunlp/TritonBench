import torch
 

def symmetric_matrix_vector_norm(
        A: torch.Tensor, x: torch.Tensor, alpha: float, beta: float, p: float=2.0) -> torch.Tensor:
    """
    Computes the matrix-vector product for a symmetric matrix `A` and a vector `x`, 
    with scaling factors `alpha` and `beta`. Then calculates the norm of the resulting vector `y`.

    Args:
        A (torch.Tensor): A symmetric matrix of shape `(n, n)`.
        x (torch.Tensor): A vector of shape `(n,)`.
        alpha (float): Scalar multiplier for the matrix-vector product.
        beta (float): Scalar multiplier added to `y`.
        p (float, optional): Order of the norm. Default is 2.0 (Euclidean norm).

    Returns:
        torch.Tensor: The norm of the resulting vector `y`.
    """
    y = alpha * torch.mv(A, x)
    y = y + beta * y
    norm = torch.norm(y, p)
    return norm

##################################################################################################################################################


import torch
torch.manual_seed(42)

def test_symmetric_matrix_vector_norm():
    results = {}

    # Test case 1: Basic test with default p value
    A = torch.tensor([[2.0, 1.0], [1.0, 2.0]], device='cuda')
    x = torch.tensor([1.0, 1.0], device='cuda')
    alpha = 1.0
    beta = 1.0
    results["test_case_1"] = symmetric_matrix_vector_norm(A, x, alpha, beta).item()

    # Test case 2: Different alpha and beta values
    alpha = 2.0
    beta = 0.5
    results["test_case_2"] = symmetric_matrix_vector_norm(A, x, alpha, beta).item()

    # Test case 3: Different p value (1-norm)
    alpha = 1.0
    beta = 1.0
    p = 1.0
    results["test_case_3"] = symmetric_matrix_vector_norm(A, x, alpha, beta, p).item()

    # Test case 4: Larger matrix and vector
    A = torch.tensor([[4.0, 1.0, 2.0], [1.0, 3.0, 1.0], [2.0, 1.0, 3.0]], device='cuda')
    x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    alpha = 1.5
    beta = 0.5
    results["test_case_4"] = symmetric_matrix_vector_norm(A, x, alpha, beta).item()

    return results

test_results = test_symmetric_matrix_vector_norm()
print(test_results)