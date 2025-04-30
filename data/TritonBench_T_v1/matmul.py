import torch

def matmul(
        tensor1: torch.Tensor, 
        tensor2: torch.Tensor) -> torch.Tensor:
    """
    Performs a matrix multiplication of two tensors.

    Args:
        tensor1 (torch.Tensor): The first tensor.
        tensor2 (torch.Tensor): The second tensor.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    return torch.matmul(tensor1, tensor2)

##################################################################################################################################################


import torch
torch.manual_seed(42)

def test_matmul():
    results = {}

    # Test case 1: Multiplying two 2D tensors
    tensor1 = torch.tensor([[1, 2], [3, 4]], device='cuda', dtype=torch.float)
    tensor2 = torch.tensor([[5, 6], [7, 8]], device='cuda', dtype=torch.float)
    results["test_case_1"] = matmul(tensor1, tensor2)

    # Test case 2: Multiplying a 1D tensor with a 2D tensor
    tensor1 = torch.tensor([1, 2], device='cuda', dtype=torch.float)
    tensor2 = torch.tensor([[3, 4], [5, 6]], device='cuda', dtype=torch.float)
    results["test_case_2"] = matmul(tensor1, tensor2)

    # Test case 3: Multiplying a 2D tensor with a 1D tensor
    tensor1 = torch.tensor([[1, 2], [3, 4]], device='cuda', dtype=torch.float)
    tensor2 = torch.tensor([5, 6], device='cuda', dtype=torch.float)
    results["test_case_3"] = matmul(tensor1, tensor2)

    # Test case 4: Multiplying two 3D tensors
    tensor1 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], device='cuda', dtype=torch.float)
    tensor2 = torch.tensor([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], device='cuda', dtype=torch.float)
    results["test_case_4"] = matmul(tensor1, tensor2)

    return results

test_results = test_matmul()
print(test_results)