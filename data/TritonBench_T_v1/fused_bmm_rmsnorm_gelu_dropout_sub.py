import torch
import torch.nn.functional as F

def fused_bmm_rmsnorm_gelu_dropout_sub(
        input1: torch.Tensor, 
        input2: torch.Tensor, 
        other: torch.Tensor, 
        normalized_shape: tuple[int], 
        dropout_p: float = 0.5, 
        training: bool = True, 
        approximate: str = 'none', 
        eps: float = 1e-05, 
        out: torch.Tensor = None) -> torch.Tensor:
    """
    Performs a fused operation combining batch matrix multiplication, RMS normalization, GELU activation, dropout, and subtraction.

    Args:
        input1 (torch.Tensor): The first input tensor.
        input2 (torch.Tensor): The second input tensor.
        other (torch.Tensor): The third input tensor to be subtracted from the output.
        normalized_shape (tuple[int]): The shape of the RMS normalization.
        dropout_p (float, optional): The dropout probability. Default: 0.5.
        training (bool, optional): Whether to apply dropout during training. Default: True.
        approximate (str, optional): The approximate method for GELU. Default: 'none'.
        eps (float, optional): The epsilon value for RMS normalization. Default: 1e-05.
        out (torch.Tensor, optional): The output tensor.

    Returns:
        torch.Tensor: The output tensor after performing the fused operation.
    """
    z1 = torch.bmm(input1, input2)
    rms_norm = F.rms_norm(z1, normalized_shape=(normalized_shape,), eps=eps)
    gelu_out = F.gelu(rms_norm, approximate=approximate)
    output = F.dropout(gelu_out, p=dropout_p, training=training) - other
    if out is not None:
        out.copy_(output)
        return out
    return output

##################################################################################################################################################


import torch
torch.manual_seed(42)

def test_fused_bmm_rmsnorm_gelu_dropout_sub():
    results = {}

    # Test case 1: Basic test with default parameters
    input1 = torch.randn(2, 3, 4, device='cuda')
    input2 = torch.randn(2, 4, 5, device='cuda')
    other = torch.randn(2, 3, 5, device='cuda')
    normalized_shape = 5
    results["test_case_1"] = fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape)

    # Test case 2: Test with different dropout probability
    dropout_p = 0.3
    results["test_case_2"] = fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, dropout_p=dropout_p)

    # Test case 3: Test with training set to False
    training = False
    results["test_case_3"] = fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, training=training)

    # Test case 4: Test with approximate GELU
    approximate = 'tanh'
    results["test_case_4"] = fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, approximate=approximate)

    return results

test_results = test_fused_bmm_rmsnorm_gelu_dropout_sub()
print(test_results)