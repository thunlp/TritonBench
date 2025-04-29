import torch
import torch.nn.functional as F
from typing import Optional

def sigmoid_batch_norm(
        input: torch.Tensor, 
        running_mean: torch.Tensor, 
        running_var: torch.Tensor, 
        weight: Optional[torch.Tensor]=None, 
        bias: Optional[torch.Tensor]=None, 
        training: bool=False, 
        momentum: float=0.1, eps: float=1e-05):
    """
    Applies Batch Normalization over the input tensor, then applies the Sigmoid activation function element-wise.

    Args:
        input (Tensor): The input tensor of shape `(N, C)` or `(N, C, L)`, where `N` is batch size, 
                         `C` is the number of features or channels, and `L` is the sequence length.
        running_mean (Tensor): The running mean of the input channels.
        running_var (Tensor): The running variance of the input channels.
        weight (Tensor, optional): Learnable scaling factor for each channel, typically represented as `γ`. Default: None.
        bias (Tensor, optional): Learnable shift for each channel, typically represented as `β`. Default: None.
        training (bool, optional): If `True`, updates running statistics; if `False`, uses them for normalization. Default: False.
        momentum (float, optional): Value for updating the running mean and variance. Default: 0.1.
        eps (float, optional): A small value added for numerical stability. Default: 1e-5.

    Returns:
        Tensor: The output tensor after batch normalization followed by the sigmoid activation.
    """
    normalized_input = F.batch_norm(input, running_mean, running_var, weight, bias, training=training, momentum=momentum, eps=eps)
    output = torch.sigmoid(normalized_input)
    return output

##################################################################################################################################################


import torch
torch.manual_seed(42)
def test_sigmoid_batch_norm():
    results = {}

    

    # Test case 1: Basic test with default parameters
    input_tensor = torch.randn(10, 5, device='cuda')
    running_mean = torch.zeros(5, device='cuda')
    running_var = torch.ones(5, device='cuda')
    results["test_case_1"] = sigmoid_batch_norm(input_tensor, running_mean, running_var)

    # Test case 2: With learnable parameters (weight and bias)
    weight = torch.ones(5, device='cuda') * 0.5
    bias = torch.zeros(5, device='cuda') + 0.1
    results["test_case_2"] = sigmoid_batch_norm(input_tensor, running_mean, running_var, weight=weight, bias=bias)

    # Test case 3: In training mode
    results["test_case_3"] = sigmoid_batch_norm(input_tensor, running_mean, running_var, training=True)

    # Test case 4: With a different momentum and eps
    results["test_case_4"] = sigmoid_batch_norm(input_tensor, running_mean, running_var, momentum=0.2, eps=1e-3)

    return results

test_results = test_sigmoid_batch_norm()
print(test_results)