import torch

def rsqrt(input: torch.Tensor, out: torch.Tensor=None) -> torch.Tensor:
    """
    Computes the reciprocal of the square root of each element in the input tensor.
    
    Args:
    - input (torch.Tensor): The input tensor.
    - out (torch.Tensor, optional): The output tensor to store the result. Default is None.
    
    Returns:
    - torch.Tensor: A tensor with the reciprocal of the square root of each element in the input.
    """
    if torch.any(input < 0):
        return torch.full_like(input, float('nan'))
    result = 1 / torch.sqrt(input)
    if out is not None:
        out.copy_(result)
        return out
    return result

##################################################################################################################################################


import torch

def test_rsqrt():
    results = {}

    # Test case 1: Positive elements
    input1 = torch.tensor([4.0, 16.0, 25.0], device='cuda')
    results["test_case_1"] = rsqrt(input1)

    # Test case 2: Contains zero
    input2 = torch.tensor([0.0, 1.0, 4.0], device='cuda')
    results["test_case_2"] = rsqrt(input2)

    # Test case 3: Contains negative elements
    input3 = torch.tensor([-1.0, 4.0, 9.0], device='cuda')
    results["test_case_3"] = rsqrt(input3)

    # Test case 4: All elements are zero
    input4 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_4"] = rsqrt(input4)

    return results

test_results = test_rsqrt()
