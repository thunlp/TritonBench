import torch
from typing import Tuple

def signbit_bitwise_and(input: torch.Tensor, other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the sign bit check and the bitwise AND operation on the input tensors.
    
    Args:
        input (Tensor): The input tensor.
        other (Tensor): The second tensor for bitwise AND, should be of integral or boolean types.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - The first tensor is a boolean tensor indicating whether the sign bit is set (True for negative values).
            - The second tensor is the result of performing bitwise AND on input and other.
    
    Example:
        >>> a = torch.tensor([0.7, -1.2, 0., 2.3])
        >>> b = torch.tensor([1, 0, 1, 1], dtype=torch.int8)
        >>> signbit_result, bitwise_and_result = signbit_bitwise_and(a, b)
        >>> signbit_result
        tensor([False, True, False, False])
        >>> bitwise_and_result
        tensor([0, 0, 0, 0], dtype=torch.int8)
    """
    signbit_result = torch.signbit(input)
    bitwise_and_result = input.to(torch.int8) & other.to(torch.int8)
    return (signbit_result, bitwise_and_result)

##################################################################################################################################################


import torch
from typing import Tuple

# def signbit_bitwise_and(input: torch.Tensor, other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     signbit_result = torch.signbit(input)
#     bitwise_and_result = input.to(torch.int8) & other.to(torch.int8)
#     return (signbit_result, bitwise_and_result)

def test_signbit_bitwise_and():
    results = {}

    # Test case 1: Positive and negative floats with integer tensor
    a = torch.tensor([0.7, -1.2, 0., 2.3], device='cuda')
    b = torch.tensor([1, 0, 1, 1], dtype=torch.int8, device='cuda')
    results["test_case_1"] = signbit_bitwise_and(a, b)

    # Test case 2: All negative floats with integer tensor
    a = torch.tensor([-0.7, -1.2, -0.1, -2.3], device='cuda')
    b = torch.tensor([1, 1, 1, 1], dtype=torch.int8, device='cuda')
    results["test_case_2"] = signbit_bitwise_and(a, b)

    # Test case 3: Mixed positive and zero floats with boolean tensor
    a = torch.tensor([0.0, 1.2, 0.0, 2.3], device='cuda')
    b = torch.tensor([True, False, True, True], dtype=torch.bool, device='cuda')
    results["test_case_3"] = signbit_bitwise_and(a, b)

    # Test case 4: All zero floats with integer tensor
    a = torch.tensor([0.0, 0.0, 0.0, 0.0], device='cuda')
    b = torch.tensor([1, 0, 1, 1], dtype=torch.int8, device='cuda')
    results["test_case_4"] = signbit_bitwise_and(a, b)

    return results

test_results = test_signbit_bitwise_and()
