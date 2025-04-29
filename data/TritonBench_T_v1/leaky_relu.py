import torch
import torch.nn.functional as F

def leaky_relu(input, negative_slope=0.01, inplace=False):
    """
    Applies the Leaky ReLU activation function element-wise to the input tensor.
    
    Args:
        input (Tensor): Input tensor.
        negative_slope (float, optional): The slope of the negative part. Default is 0.01.
        inplace (bool, optional): If set to True, will modify the input tensor in place. Default is False.
        
    Returns:
        Tensor: A tensor with the Leaky ReLU function applied element-wise.
    """
    return F.leaky_relu(input, negative_slope=negative_slope, inplace=inplace)

##################################################################################################################################################


import torch

def test_leaky_relu():
    results = {}

    # Test case 1: Default parameters
    input_tensor_1 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_1"] = leaky_relu(input_tensor_1)

    # Test case 2: Custom negative_slope
    input_tensor_2 = torch.tensor([-2.0, 0.0, 2.0], device='cuda')
    results["test_case_2"] = leaky_relu(input_tensor_2, negative_slope=0.1)

    # Test case 3: Inplace operation
    input_tensor_3 = torch.tensor([-3.0, 0.0, 3.0], device='cuda')
    results["test_case_3"] = leaky_relu(input_tensor_3, inplace=True)

    # Test case 4: Larger tensor
    input_tensor_4 = torch.tensor([-4.0, -2.0, 0.0, 2.0, 4.0], device='cuda')
    results["test_case_4"] = leaky_relu(input_tensor_4)

    return results

test_results = test_leaky_relu()
print(test_results)