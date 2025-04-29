import torch

def logit(input: torch.Tensor, eps: float=None, out: torch.Tensor=None) -> torch.Tensor:
    """
    Computes the logit of the elements of the input tensor.

    Args:
        input (Tensor): The input tensor, should be in the range [0, 1].
        eps (float, optional): The epsilon for clamping the input. Defaults to None.
        out (Tensor, optional): The output tensor. Defaults to None.

    Returns:
        Tensor: A new tensor with the logit of the elements of the input tensor.
    """
    if eps is not None:
        input = input.clamp(eps, 1 - eps)
    return torch.log(input / (1 - input), out=out)

##################################################################################################################################################


import torch

def test_logit():
    results = {}

    # Test case 1: Basic test with input tensor in range [0, 1] without eps
    input1 = torch.tensor([0.2, 0.5, 0.8], device='cuda')
    results["test_case_1"] = logit(input1)

    # Test case 2: Test with input tensor in range [0, 1] with eps
    input2 = torch.tensor([0.0, 0.5, 1.0], device='cuda')
    eps = 1e-6
    results["test_case_2"] = logit(input2, eps=eps)

    # Test case 3: Test with input tensor in range [0, 1] with eps and out tensor
    input3 = torch.tensor([0.1, 0.9], device='cuda')
    out = torch.empty_like(input3)
    results["test_case_3"] = logit(input3, eps=eps, out=out)

    # Test case 4: Test with input tensor in range [0, 1] with out tensor
    input4 = torch.tensor([0.3, 0.7], device='cuda')
    out = torch.empty_like(input4)
    results["test_case_4"] = logit(input4, out=out)

    return results

test_results = test_logit()
print(test_results)