import torch
import torch.nn.functional as F
from typing import Optional

def fused_mv_logsoftmax_dropout(
        input: torch.Tensor, 
        vec: torch.Tensor, 
        p: float=0.5, 
        training: bool=True, 
        inplace: bool=False, 
        dim: int=0, 
        out: Optional[torch.Tensor]=None) -> torch.Tensor:
    """
    Performs a fused operation combining matrix-vector multiplication, log-softmax activation, and dropout.
    
    Args:
        input (Tensor): The input matrix of shape (n, m).
        vec (Tensor): The vector of shape (m).
        p (float, optional): The probability of an element to be zeroed in dropout. Default is 0.5.
        training (bool, optional): If True, dropout is applied. If False, dropout is not applied. Default is True.
        inplace (bool, optional): If True, the operation is done in place. Default is False.
        dim (int, optional): The dimension along which the log-softmax will be computed. Default is 0.
        out (Tensor, optional): A tensor to store the result. If not specified, a new tensor is returned.
    
    Returns:
        Tensor: The result after matrix-vector multiplication, log-softmax, and dropout.
    """
    z = torch.mv(input, vec)
    s = F.log_softmax(z, dim=dim)
    y = F.dropout(s, p=p, training=training, inplace=inplace)
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################


import torch

def test_fused_mv_logsoftmax_dropout():
    results = {}

    # Test case 1: Basic functionality
    input1 = torch.randn(3, 4, device='cuda')
    vec1 = torch.randn(4, device='cuda')
    results["test_case_1"] = fused_mv_logsoftmax_dropout(input1, vec1)

    # Test case 2: Dropout with p=0.2
    input2 = torch.randn(3, 4, device='cuda')
    vec2 = torch.randn(4, device='cuda')
    results["test_case_2"] = fused_mv_logsoftmax_dropout(input2, vec2, p=0.2)

    # Test case 3: Dropout in evaluation mode (training=False)
    input3 = torch.randn(3, 4, device='cuda')
    vec3 = torch.randn(4, device='cuda')
    results["test_case_3"] = fused_mv_logsoftmax_dropout(input3, vec3, training=False)

    # Test case 4: Inplace operation
    input4 = torch.randn(3, 4, device='cuda')
    vec4 = torch.randn(4, device='cuda')
    results["test_case_4"] = fused_mv_logsoftmax_dropout(input4, vec4, inplace=True)

    return results

test_results = test_fused_mv_logsoftmax_dropout()
print(test_results)