import torch

def broadcast_tensors(
        x: torch.Tensor, 
        y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    This function broadcasts two tensors to the same shape.
    Args:
        x (torch.Tensor): The first tensor.
        y (torch.Tensor): The second tensor.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The broadcasted tensors.

    Example:
        b_x, b_y = broadcast_tensors(torch.tensor(1.0), torch.tensor([1.0, 5.0]))
        print(b_x)  # Output: tensor([1., 1.])
        print(b_y)  # Output: tensor([1., 5.])
    """
    (a, b) = torch.broadcast_tensors(x, y)
    return (a, b)

##################################################################################################################################################

import torch
torch.manual_seed(42)

def test_broadcast_tensors():
    results = {}

    # Test case 1: Broadcasting a scalar and a 1D tensor
    x1 = torch.tensor(3.0, device='cuda')
    y1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_1"] = broadcast_tensors(x1, y1)

    # Test case 2: Broadcasting two 1D tensors of different sizes
    x2 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    y2 = torch.tensor([1.0], device='cuda')
    results["test_case_2"] = broadcast_tensors(x2, y2)

    # Test case 3: Broadcasting a 2D tensor and a 1D tensor
    x3 = torch.tensor([[1.0, 2.0, 3.0]], device='cuda')
    y3 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_3"] = broadcast_tensors(x3, y3)

    # Test case 4: Broadcasting two 2D tensors of different shapes
    x4 = torch.tensor([[1.0], [2.0], [3.0]], device='cuda')
    y4 = torch.tensor([[1.0, 2.0, 3.0]], device='cuda')
    results["test_case_4"] = broadcast_tensors(x4, y4)

    return results

test_results = test_broadcast_tensors()
print(test_results)