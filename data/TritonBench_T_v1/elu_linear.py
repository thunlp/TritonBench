import torch
import torch.nn.functional as F


def elu_linear(input, weight, bias=None, alpha=1.0, inplace=False):
    """
    Applies a linear transformation followed by the Exponential Linear Unit (ELU) activation.

    Parameters:
        input (Tensor): The input tensor for the linear layer. 
                         Shape should be (batch_size, in_features).
        weight (Tensor): The weight tensor for the linear transformation.
                         Shape should be (out_features, in_features).
        bias (Tensor, optional): The bias tensor for the linear transformation. Default: None.
                                  Shape should be (out_features).
        alpha (float, optional): The α parameter for the ELU function. Default: 1.0.
        inplace (bool, optional): Whether to apply ELU in-place. Default: False.

    Returns:
        Tensor: The output tensor after applying the linear transformation and ELU activation.
                Shape will be (batch_size, out_features).
    """
    output = F.linear(input, weight, bias)
    return F.elu(output, alpha=alpha, inplace=inplace)

##################################################################################################################################################


import torch
import torch.nn.functional as F

# def elu_linear(input, weight, bias=None, alpha=1.0, inplace=False):
#     output = F.linear(input, weight, bias)
#     return F.elu(output, alpha=alpha, inplace=inplace)

def test_elu_linear():
    results = {}

    # Test case 1: Basic test with bias, alpha=1.0, inplace=False
    input1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight1 = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], device='cuda')
    bias1 = torch.tensor([0.0, 0.0], device='cuda')
    results["test_case_1"] = elu_linear(input1, weight1, bias1)

    # Test case 2: Without bias, alpha=1.0, inplace=False
    input2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight2 = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], device='cuda')
    results["test_case_2"] = elu_linear(input2, weight2)

    # Test case 3: With bias, alpha=0.5, inplace=False
    input3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight3 = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], device='cuda')
    bias3 = torch.tensor([0.0, 0.0], device='cuda')
    results["test_case_3"] = elu_linear(input3, weight3, bias3, alpha=0.5)

    # Test case 4: With bias, alpha=1.0, inplace=True
    input4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight4 = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], device='cuda')
    bias4 = torch.tensor([0.0, 0.0], device='cuda')
    results["test_case_4"] = elu_linear(input4, weight4, bias4, inplace=True)

    return results

test_results = test_elu_linear()
