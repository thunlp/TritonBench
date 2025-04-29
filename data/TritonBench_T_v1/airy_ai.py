import torch


def airy_ai(input: torch.Tensor, out: torch.Tensor=None) -> torch.Tensor:
    """
    Computes the Airy function Ai for each element of the input tensor.

    Args:
        input (Tensor): The input tensor.
        out (Tensor, optional): The output tensor. If provided, the result will be stored in this tensor.

    Returns:
        Tensor: A tensor containing the values of the Airy function Ai for each element of the input tensor.
    """
    return torch.special.airy_ai(input)

##################################################################################################################################################


import torch
torch.manual_seed(42)

def test_airy_ai():
    results = {}

    # Test case 1: Single positive value
    input1 = torch.tensor([1.0], device='cuda')
    results["test_case_1"] = airy_ai(input1)

    # Test case 2: Single negative value
    input2 = torch.tensor([-1.0], device='cuda')
    results["test_case_2"] = airy_ai(input2)

    # Test case 3: Tensor with multiple values
    input3 = torch.tensor([0.0, 1.0, -1.0], device='cuda')
    results["test_case_3"] = airy_ai(input3)

    # Test case 4: Tensor with large positive and negative values
    input4 = torch.tensor([10.0, -10.0], device='cuda')
    results["test_case_4"] = airy_ai(input4)

    return results

test_results = test_airy_ai()
print(test_results)