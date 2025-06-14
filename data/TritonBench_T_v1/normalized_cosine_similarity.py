import torch
import torch.nn.functional as F
from torch import Tensor

def normalized_cosine_similarity(x1: Tensor, x2: Tensor, dim: int=1, eps_similarity: float=1e-08, p_norm: float=2, eps_norm: float=1e-12) -> Tensor:
    x1_normalized = torch.nn.functional.normalize(x1, p=p_norm, dim=dim, eps=eps_norm)
    x2_normalized = torch.nn.functional.normalize(x2, p=p_norm, dim=dim, eps=eps_norm)
    return torch.nn.functional.cosine_similarity(x1_normalized, x2_normalized, dim=dim, eps=eps_similarity)

##################################################################################################################################################


import torch
import torch.nn.functional as F
from torch import Tensor

# def normalized_cosine_similarity(x1: Tensor, x2: Tensor, dim: int=1, eps_similarity: float=1e-08, p_norm: float=2, eps_norm: float=1e-12) -> Tensor:
#     x1_normalized = torch.nn.functional.normalize(x1, p=p_norm, dim=dim, eps=eps_norm)
#     x2_normalized = torch.nn.functional.normalize(x2, p=p_norm, dim=dim, eps=eps_norm)
#     return torch.nn.functional.cosine_similarity(x1_normalized, x2_normalized, dim=dim, eps=eps_similarity)

def test_normalized_cosine_similarity():
    results = {}

    # Test case 1: Basic test with default parameters
    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    x2 = torch.tensor([[2.0, 3.0], [4.0, 5.0]], device='cuda')
    results["test_case_1"] = normalized_cosine_similarity(x1, x2)

    # Test case 2: Different dimension
    x1 = torch.tensor([[1.0, 2.0, 3.0]], device='cuda')
    x2 = torch.tensor([[2.0, 3.0, 4.0]], device='cuda')
    results["test_case_2"] = normalized_cosine_similarity(x1, x2, dim=0)

    # Test case 3: Different p_norm
    x1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    x2 = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device='cuda')
    results["test_case_3"] = normalized_cosine_similarity(x1, x2, p_norm=1)

    # Test case 4: Different eps_norm
    x1 = torch.tensor([[1e-10, 0.0], [0.0, 1e-10]], device='cuda')
    x2 = torch.tensor([[0.0, 1e-10], [1e-10, 0.0]], device='cuda')
    results["test_case_4"] = normalized_cosine_similarity(x1, x2, eps_norm=1e-10)

    return results

test_results = test_normalized_cosine_similarity()
