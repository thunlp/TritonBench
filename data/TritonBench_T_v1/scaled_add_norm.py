import torch

def scaled_add_norm(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Modifies the target tensor `y` in place by adding `alpha * x` to it.
    Then computes and returns the 2-norm of the modified `y`.

    Args:
        y (torch.Tensor): The target tensor to be modified, of shape (n,).
        x (torch.Tensor): The tensor to be scaled and added to `y`, of shape (n,).
        alpha (float): The scalar multiplier for `x`.

    Returns:
        torch.Tensor: The 2-norm (Euclidean norm) of the updated `y`.
    """
    y += alpha * x
    return torch.norm(y)

##################################################################################################################################################


import torch
torch.manual_seed(42)

def test_scaled_add_norm():
    results = {}

    # Test case 1: Basic test with small tensors
    y1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    x1 = torch.tensor([0.5, 0.5, 0.5], device='cuda')
    alpha1 = 2.0
    results["test_case_1"] = scaled_add_norm(y1, x1, alpha1).item()

    # Test case 2: Test with negative alpha
    y2 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    x2 = torch.tensor([0.5, 0.5, 0.5], device='cuda')
    alpha2 = -1.0
    results["test_case_2"] = scaled_add_norm(y2, x2, alpha2).item()

    # Test case 3: Test with zero alpha
    y3 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    x3 = torch.tensor([0.5, 0.5, 0.5], device='cuda')
    alpha3 = 0.0
    results["test_case_3"] = scaled_add_norm(y3, x3, alpha3).item()

    # Test case 4: Test with zero vector x
    y4 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    x4 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    alpha4 = 2.0
    results["test_case_4"] = scaled_add_norm(y4, x4, alpha4).item()

    return results

test_results = test_scaled_add_norm()
print(test_results)