import torch
import torch.nn.functional as F
from typing import Optional

def fused_mv_sigmoid_sub(
        input: torch.Tensor, 
        vec: torch.Tensor, 
        other: torch.Tensor, 
        alpha: float=1,
        out: Optional[torch.Tensor]=None) -> torch.Tensor:
    """
    Performs a fused operation combining matrix-vector multiplication, sigmoid activation, and subtraction.

    Args:
        input (Tensor): Input matrix A of shape (n, m).
        vec (Tensor): Input vector v of shape (m).
        other (Tensor or Number): Tensor or scalar b to subtract from the sigmoid output, scaled by alpha.
        alpha (Number, optional): Scalar multiplier for other. Default: 1.
        out (Tensor, optional): Output tensor. Ignored if None. Default: None.

    Returns:
        Tensor: The result of the fused operation.
    """
    z = torch.mv(input, vec)
    s = torch.sigmoid(z)
    y = torch.sub(s, other, alpha=alpha)
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################


import torch
torch.manual_seed(42)

def test_fused_mv_sigmoid_sub():
    results = {}
    
    # Test case 1: Basic functionality
    input1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    vec1 = torch.tensor([1.0, 1.0], device='cuda')
    other1 = torch.tensor([0.5, 0.5], device='cuda')
    results["test_case_1"] = fused_mv_sigmoid_sub(input1, vec1, other1)
    
    # Test case 2: Scalar other
    input2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    vec2 = torch.tensor([1.0, 1.0], device='cuda')
    other2 = 0.5
    results["test_case_2"] = fused_mv_sigmoid_sub(input2, vec2, other2)
    
    # Test case 3: Different alpha
    input3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    vec3 = torch.tensor([1.0, 1.0], device='cuda')
    other3 = torch.tensor([0.5, 0.5], device='cuda')
    results["test_case_3"] = fused_mv_sigmoid_sub(input3, vec3, other3, alpha=2)
    
    # Test case 4: Output tensor provided
    input4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    vec4 = torch.tensor([1.0, 1.0], device='cuda')
    other4 = torch.tensor([0.5, 0.5], device='cuda')
    out4 = torch.empty(2, device='cuda')
    results["test_case_4"] = fused_mv_sigmoid_sub(input4, vec4, other4, out=out4)
    
    return results

test_results = test_fused_mv_sigmoid_sub()
print(test_results)