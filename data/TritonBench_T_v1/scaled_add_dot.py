import torch

def scaled_add_dot(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Performs two operations in a single function:
    1. Scales the tensor `x` by the factor `alpha` and adds it to `y`.
    2. Computes the dot product of the modified `y` with itself.

    Args:
        y (Tensor): The target tensor to be modified, of shape (n,).
        x (Tensor): The tensor to be scaled and added to `y`, of shape (n,).
        alpha (float): The scalar multiplier for `x`.

    Returns:
        Tensor: The dot product of the modified `y` with itself.
    """
    y += alpha * x
    return torch.dot(y, y)

##################################################################################################################################################


import torch

def test_scaled_add_dot():
    results = {}

    # Test case 1: Basic functionality
    y1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    x1 = torch.tensor([0.5, 0.5, 0.5], device='cuda')
    alpha1 = 2.0
    results["test_case_1"] = scaled_add_dot(y1, x1, alpha1).item()

    # Test case 2: Zero tensor x
    y2 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    x2 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    alpha2 = 2.0
    results["test_case_2"] = scaled_add_dot(y2, x2, alpha2).item()

    # Test case 3: Zero tensor y
    y3 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    x3 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    alpha3 = 1.0
    results["test_case_3"] = scaled_add_dot(y3, x3, alpha3).item()

    # Test case 4: Negative alpha
    y4 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    x4 = torch.tensor([0.5, 0.5, 0.5], device='cuda')
    alpha4 = -1.0
    results["test_case_4"] = scaled_add_dot(y4, x4, alpha4).item()

    return results

test_results = test_scaled_add_dot()
print(test_results)