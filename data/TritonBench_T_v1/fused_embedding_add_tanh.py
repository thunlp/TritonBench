import torch
import torch.nn.functional as F

def fused_embedding_add_tanh(
        input_indices: torch.Tensor , 
        weight: torch.Tensor, 
        other: torch.Tensor, 
        *, 
        padding_idx: int=None, 
        max_norm: float=None, 
        norm_type: float=2.0, 
        scale_grad_by_freq: bool=False, 
        sparse: bool=False, 
        out: torch.Tensor=None) -> torch.Tensor:
    """
    Computes the fused embedding and adds the other tensor to it, then applies the tanh function.
    
    Args:
        input_indices (torch.Tensor): The input indices.
        weight (torch.Tensor): The weight tensor.
        other (torch.Tensor): The other tensor.
        padding_idx (int, optional): The padding index.
        max_norm (float, optional): The max norm.
        norm_type (float, optional): The norm type.
        scale_grad_by_freq (bool, optional): The scale grad by freq.
        sparse (bool, optional): The sparse.
        out (torch.Tensor, optional): The output tensor.

    Returns:
        torch.Tensor: The result.   
    """
    embeddings = F.embedding(input_indices, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
    sum_embeddings = embeddings + other
    result = torch.tanh(sum_embeddings)
    if out is not None:
        out.copy_(result)
    return result

##################################################################################################################################################


import torch

def test_fused_embedding_add_tanh():
    results = {}

    # Test case 1: Basic test without padding_idx, max_norm, scale_grad_by_freq, sparse, and out
    input_indices = torch.tensor([1, 2, 3], device='cuda')
    weight = torch.randn(5, 3, device='cuda')
    other = torch.randn(3, 3, device='cuda')
    results["test_case_1"] = fused_embedding_add_tanh(input_indices, weight, other)

    # Test case 2: Test with padding_idx
    padding_idx = 0
    input_indices = torch.tensor([0, 1, 2], device='cuda')
    weight = torch.randn(5, 3, device='cuda')
    other = torch.randn(3, 3, device='cuda')
    results["test_case_2"] = fused_embedding_add_tanh(input_indices, weight, other, padding_idx=padding_idx)

    # Test case 3: Test with max_norm
    max_norm = 1.0
    input_indices = torch.tensor([1, 2, 3], device='cuda')
    weight = torch.randn(5, 3, device='cuda')
    other = torch.randn(3, 3, device='cuda')
    results["test_case_3"] = fused_embedding_add_tanh(input_indices, weight, other, max_norm=max_norm)

    # Test case 4: Test with norm_type
    norm_type = 1.0
    input_indices = torch.tensor([1, 2, 3], device='cuda')
    weight = torch.randn(5, 3, device='cuda')
    other = torch.randn(3, 3, device='cuda')
    results["test_case_4"] = fused_embedding_add_tanh(input_indices, weight, other, norm_type=norm_type)

    return results

test_results = test_fused_embedding_add_tanh()
print(test_results)