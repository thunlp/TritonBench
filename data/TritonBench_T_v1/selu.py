import torch

def selu(input: torch.Tensor, inplace: bool=False) -> torch.Tensor:
    """
    Applies the element-wise SELU (Scaled Exponential Linear Unit) function to the input tensor.

    The SELU function is defined as:
    SELU(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
    where alpha is approximately 1.673 and scale is approximately 1.051.

    Args:
    - input (torch.Tensor): The input tensor.
    - inplace (bool, optional): If set to True, will do the operation in-place. Default is False.

    Returns:
    - torch.Tensor: The resulting tensor after applying SELU function.
    """
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    positive_part = torch.maximum(input, torch.zeros_like(input))

    negative_part_calc = alpha * (torch.exp(input) - 1)
    negative_part = torch.minimum(torch.zeros_like(input), negative_part_calc)

    result = scale * (positive_part + negative_part)

    if inplace:
        input.copy_(result)
        return input
    else:
        return result

##################################################################################################################################################


def test_selu():
    # Initialize a dictionary to store test results
    results = {}

    # Test case 1: Positive values
    input_tensor_1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_1"] = selu(input_tensor_1)

    # Test case 2: Negative values
    input_tensor_2 = torch.tensor([-1.0, -2.0, -3.0], device='cuda')
    results["test_case_2"] = selu(input_tensor_2)

    # Test case 3: Mixed values
    input_tensor_3 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_3"] = selu(input_tensor_3)

    # Test case 4: Zero values
    input_tensor_4 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_4"] = selu(input_tensor_4)

    return results

test_results = test_selu()
print(test_results)