import torch
import torch.special


def sigmoid(input, out=None):
    """
    Applies the Sigmoid function element-wise on the input tensor.
    
    The sigmoid function is defined as:
        sigmoid(x) = 1 / (1 + exp(-x))
    
    Args:
        input (Tensor): The input tensor.
        out (Tensor, optional): The output tensor.

    Returns:
        Tensor: A tensor with the sigmoid function applied element-wise.
    """
    return torch.special.expit(input, out=out)

##################################################################################################################################################


import torch
import torch.special

def test_sigmoid():
    results = {}

    # Test case 1: Simple tensor on GPU
    input_tensor_1 = torch.tensor([0.0, 1.0, -1.0], device='cuda')
    results["test_case_1"] = sigmoid(input_tensor_1)

    # Test case 2: Larger tensor with positive and negative values on GPU
    input_tensor_2 = torch.tensor([0.5, -0.5, 2.0, -2.0], device='cuda')
    results["test_case_2"] = sigmoid(input_tensor_2)

    # Test case 3: 2D tensor on GPU
    input_tensor_3 = torch.tensor([[0.0, 1.0], [-1.0, 2.0]], device='cuda')
    results["test_case_3"] = sigmoid(input_tensor_3)

    # Test case 4: Tensor with all zeros on GPU
    input_tensor_4 = torch.zeros(3, 3, device='cuda')
    results["test_case_4"] = sigmoid(input_tensor_4)

    return results

test_results = test_sigmoid()
