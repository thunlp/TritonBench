import torch
import torch.nn.functional as F
from typing import Optional

def fused_masked_select_add_gelu(
        input: torch.Tensor, 
        mask: torch.Tensor, 
        other: torch.Tensor, 
        *, 
        alpha: float=1, 
        approximate: str='none', 
        out: Optional[torch.Tensor]=None) -> torch.Tensor:
    """
    Perform a fused operation combining masked selection, addition, and GELU activation.
    
    Parameters:
    - input (Tensor): The input tensor.
    - mask (Tensor): A boolean tensor of the same shape as input.
    - other (Tensor or Scalar): The value to add to the selected elements.
    - alpha (float, optional): A scaling factor for `other`. Default is 1.
    - approximate (str, optional): The approximation method for GELU ('none' or 'tanh'). Default is 'none'.
    - out (Tensor, optional): The output tensor to store the result. Default is None.
    
    Returns:
    - Tensor: The result tensor after the fused operation.
    """

    Z = torch.masked_select(input, mask)
    S = torch.add(Z, other, alpha=alpha)
    Y = F.gelu(S, approximate=approximate)
    if out is not None:
        out.copy_(Y)
        return out
    return Y

##################################################################################################################################################


import torch
import torch.nn.functional as F


def test_fused_masked_select_add_gelu():
    results = {}
    
    # Test case 1: Basic test with default parameters
    input1 = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
    mask1 = torch.tensor([True, False, True, False], device='cuda')
    other1 = 1.0
    results["test_case_1"] = fused_masked_select_add_gelu(input1, mask1, other1)
    
    # Test case 2: Test with alpha parameter
    input2 = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
    mask2 = torch.tensor([True, True, False, False], device='cuda')
    other2 = 2.0
    results["test_case_2"] = fused_masked_select_add_gelu(input2, mask2, other2, alpha=0.5)
    
    # Test case 3: Test with approximate='tanh'
    input3 = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
    mask3 = torch.tensor([False, True, True, False], device='cuda')
    other3 = 1.0
    results["test_case_3"] = fused_masked_select_add_gelu(input3, mask3, other3, approximate='tanh')
    
    # Test case 4: Test with out parameter
    input4 = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
    mask4 = torch.tensor([True, False, True, True], device='cuda')
    other4 = 1.0
    out4 = torch.empty(3, device='cuda')
    results["test_case_4"] = fused_masked_select_add_gelu(input4, mask4, other4, out=out4)
    
    return results

test_results = test_fused_masked_select_add_gelu()
print(test_results)