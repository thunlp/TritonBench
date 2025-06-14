import torch
import torch.nn.functional as F

def fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape, dropout_p=0.1, eps=1e-05, training=True, approximate='none', *, out=None):
    z1 = torch.bmm(input1, input2)
    rms_norm = F.rms_norm(z1, normalized_shape=(normalized_shape,), eps=eps)
    gelu_out = F.gelu(rms_norm, approximate=approximate)
    output = F.dropout(gelu_out, p=dropout_p, training=training)
    if out is not None:
        out.copy_(output)
        return out
    return output

##################################################################################################################################################


import torch
import torch.nn.functional as F

# def fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape, dropout_p=0.1, eps=1e-05, training=True, approximate='none', *, out=None):
#     z1 = torch.bmm(input1, input2)
#     rms_norm = F.rms_norm(z1, normalized_shape=(normalized_shape,), eps=eps)
#     gelu_out = F.gelu(rms_norm, approximate=approximate)
#     output = F.dropout(gelu_out, p=dropout_p, training=training)
#     if out is not None:
#         out.copy_(output)
#         return out
#     return output

def test_fused_bmm_rmsnorm_gelu_dropout():
    results = {}
    
    # Test case 1: Default parameters
    input1 = torch.randn(2, 3, 4, device='cuda')
    input2 = torch.randn(2, 4, 5, device='cuda')
    results["test_case_1"] = fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape=5)
    
    # Test case 2: Different dropout probability
    results["test_case_2"] = fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape=5, dropout_p=0.2)
    
    # Test case 3: Non-training mode
    results["test_case_3"] = fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape=5, training=False)
    
    # Test case 4: Different approximation method for GELU
    results["test_case_4"] = fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape=5, approximate='tanh')
    
    return results

test_results = test_fused_bmm_rmsnorm_gelu_dropout()
