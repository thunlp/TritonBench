import torch
import torch.nn.functional as F

def fused_bmm_dropout_gelu(input1, input2, p=0.5, training=True, inplace=False, approximate='none', *, out=None):
    """
    Performs a fused operation combining batch matrix multiplication, dropout, and GELU activation.

    Args:
        input1 (Tensor): First input tensor for batch matrix multiplication, of shape (B, N, M).
        input2 (Tensor): Second input tensor for batch matrix multiplication, of shape (B, M, P).
        p (float, optional): Probability of an element to be zeroed in the dropout layer. Default: 0.5.
        training (bool, optional): Apply dropout if True. Default: True.
        inplace (bool, optional): If True, will perform the dropout operation in-place. Default: False.
        approximate (str, optional): The approximation to use for GELU. Default: 'none'. Can be 'none' or 'tanh'.
        out (Tensor, optional): Output tensor to store the result. If None, a new tensor is returned.

    Returns:
        Tensor: The output tensor after performing batch matrix multiplication, dropout, and GELU activation.
    """
    Z = torch.bmm(input1, input2)
    D = torch.nn.functional.dropout(Z, p=p, training=training, inplace=inplace)
    O = torch.nn.functional.gelu(D, approximate=approximate)
    if out is not None:
        out.copy_(O)
        return out
    return O

##################################################################################################################################################


import torch
import torch.nn.functional as F

# def fused_bmm_dropout_gelu(input1, input2, p=0.5, training=True, inplace=False, approximate='none', *, out=None):
#     Z = torch.bmm(input1, input2)
#     D = torch.nn.functional.dropout(Z, p=p, training=training, inplace=inplace)
#     O = torch.nn.functional.gelu(D, approximate=approximate)
#     if out is not None:
#         out.copy_(O)
#         return out
#     return O

def test_fused_bmm_dropout_gelu():
    results = {}
    
    # Test case 1: Default parameters
    input1 = torch.randn(2, 3, 4, device='cuda')
    input2 = torch.randn(2, 4, 5, device='cuda')
    results["test_case_1"] = fused_bmm_dropout_gelu(input1, input2)
    
    # Test case 2: Dropout with p=0.3 and training=False
    input1 = torch.randn(2, 3, 4, device='cuda')
    input2 = torch.randn(2, 4, 5, device='cuda')
    results["test_case_2"] = fused_bmm_dropout_gelu(input1, input2, p=0.3, training=False)
    
    # Test case 3: In-place dropout
    input1 = torch.randn(2, 3, 4, device='cuda')
    input2 = torch.randn(2, 4, 5, device='cuda')
    results["test_case_3"] = fused_bmm_dropout_gelu(input1, input2, inplace=True)
    
    # Test case 4: GELU with tanh approximation
    input1 = torch.randn(2, 3, 4, device='cuda')
    input2 = torch.randn(2, 4, 5, device='cuda')
    results["test_case_4"] = fused_bmm_dropout_gelu(input1, input2, approximate='tanh')
    
    return results

test_results = test_fused_bmm_dropout_gelu()
