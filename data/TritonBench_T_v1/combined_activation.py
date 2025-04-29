import torch


def combined_activation(
        input: torch.Tensor, 
        weight1: torch.Tensor, 
        weight2: torch.Tensor, 
        bias: torch.Tensor, 
        *, 
        out: torch.Tensor = None) -> torch.Tensor:
    """
    Perform the combined activation function which includes matrix multiplication,
    sigmoid, tanh, element-wise multiplication, and addition.

    Args:
        input (Tensor): Input tensor of shape (*, N, D_in), where * denotes any batch dimensions.
        weight1 (Tensor): Weight matrix of shape (D_in, D_out).
        weight2 (Tensor): Weight tensor for element-wise multiplication, must be broadcastable 
                          to the shape of the intermediate activation.
        bias (Tensor): Bias tensor, must be broadcastable to the shape of the output.
        out (Tensor, optional): Output tensor to store the result, ignored if None.

    Returns:
        Tensor: Output tensor of shape (*, N, D_out).
    """
    z = torch.mm(input, weight1)
    s = torch.sigmoid(z)
    t = torch.tanh(s)
    m = t * weight2
    y = m + bias
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################


import torch
torch.manual_seed(42)

def test_combined_activation():
    results = {}

    # Test case 1
    input1 = torch.randn(2, 3, device='cuda')
    weight1_1 = torch.randn(3, 4, device='cuda')
    weight2_1 = torch.randn(2, 4, device='cuda')
    bias1 = torch.randn(2, 4, device='cuda')
    results["test_case_1"] = combined_activation(input1, weight1_1, weight2_1, bias1)

    # Test case 2
    input2 = torch.randn(3, 3, device='cuda')
    weight1_2 = torch.randn(3, 5, device='cuda')
    weight2_2 = torch.randn(3, 5, device='cuda')
    bias2 = torch.randn(3, 5, device='cuda')
    results["test_case_2"] = combined_activation(input2, weight1_2, weight2_2, bias2)

    # Test case 3
    input3 = torch.randn(4, 3, device='cuda')
    weight1_3 = torch.randn(3, 6, device='cuda')
    weight2_3 = torch.randn(4, 6, device='cuda')
    bias3 = torch.randn(4, 6, device='cuda')
    results["test_case_3"] = combined_activation(input3, weight1_3, weight2_3, bias3)

    # Test case 4
    input4 = torch.randn(5, 3, device='cuda')
    weight1_4 = torch.randn(3, 7, device='cuda')
    weight2_4 = torch.randn(5, 7, device='cuda')
    bias4 = torch.randn(5, 7, device='cuda')
    results["test_case_4"] = combined_activation(input4, weight1_4, weight2_4, bias4)

    return results

test_results = test_combined_activation()
print(test_results)