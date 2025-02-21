import torch
import torch.nn.functional as F

def bitwise_and_binomial(input: torch.Tensor, other: torch.Tensor, total_count: torch.Tensor, probs: torch.Tensor=None, logits: torch.Tensor=None) -> torch.Tensor:
    """
    Computes the bitwise AND operation between two tensors and then applies a Binomial distribution sampling based on the resulting tensor's values.
    
    Arguments:
    - input (Tensor): The first input tensor of integral or Boolean type.
    - other (Tensor): The second input tensor of integral or Boolean type.
    - total_count (Tensor): Number of Bernoulli trials, must be broadcastable with `probs` or `logits`.
    - probs (Tensor, optional): Event probabilities. Only one of `probs` or `logits` should be provided.
    - logits (Tensor, optional): Event log-odds.
    
    Returns:
    - Tensor: The output tensor resulting from the Binomial distribution applied to the bitwise AND results.
    """
    bitwise_and_result = input & other
    if probs is not None:
        return torch.distributions.Binomial(total_count=total_count, probs=probs).sample()
    elif logits is not None:
        probs_from_logits = torch.sigmoid(logits)
        return torch.distributions.Binomial(total_count=total_count, probs=probs_from_logits).sample()
    else:
        raise ValueError('Either `probs` or `logits` must be provided for Binomial distribution.')

##################################################################################################################################################


import torch
import torch.nn.functional as F

def test_bitwise_and_binomial():
    results = {}

    # Test case 1: Using `probs`
    input_tensor = torch.tensor([1, 0, 1, 0], dtype=torch.int32, device='cuda')
    other_tensor = torch.tensor([1, 1, 0, 0], dtype=torch.int32, device='cuda')
    total_count = torch.tensor([5, 5, 5, 5], dtype=torch.float32, device='cuda')
    probs = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32, device='cuda')
    results["test_case_1"] = bitwise_and_binomial(input_tensor, other_tensor, total_count, probs=probs)

    # Test case 2: Using `logits`
    logits = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device='cuda')
    results["test_case_2"] = bitwise_and_binomial(input_tensor, other_tensor, total_count, logits=logits)

    # Test case 3: Different `total_count` with `probs`
    total_count_diff = torch.tensor([10, 10, 10, 10], dtype=torch.float32, device='cuda')
    results["test_case_3"] = bitwise_and_binomial(input_tensor, other_tensor, total_count_diff, probs=probs)

    # Test case 4: Different `total_count` with `logits`
    results["test_case_4"] = bitwise_and_binomial(input_tensor, other_tensor, total_count_diff, logits=logits)

    return results

test_results = test_bitwise_and_binomial()
