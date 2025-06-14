import torch
import math
from typing import Tuple

def erfc_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the complementary error function (erfc) and the square root of each element in the input tensor.

    Args:
        input (torch.Tensor): The input tensor for which the erfc and square root are computed.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - erfc_result (torch.Tensor): The complementary error function results.
            - sqrt_result (torch.Tensor): The square root results.
    """
    erfc_result = torch.erfc(input)
    sqrt_result = torch.sqrt(input)
    sqrt_result[input < 0] = float('nan')
    return (erfc_result, sqrt_result)

##################################################################################################################################################


import torch
import math
from typing import Tuple

# def erfc_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Computes the complementary error function (erfc) and the square root of each element in the input tensor.

#     Args:
#         input (torch.Tensor): The input tensor for which the erfc and square root are computed.

#     Returns:
#         Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
#             - erfc_result (torch.Tensor): The complementary error function results.
#             - sqrt_result (torch.Tensor): The square root results.
#     """
#     erfc_result = torch.erfc(input)
#     sqrt_result = torch.sqrt(input)
#     sqrt_result[input < 0] = float('nan')
#     return (erfc_result, sqrt_result)

def test_erfc_sqrt():
    results = {}

    # Test case 1: Positive values
    input1 = torch.tensor([0.0, 1.0, 2.0], device='cuda')
    results["test_case_1"] = erfc_sqrt(input1)

    # Test case 2: Negative values
    input2 = torch.tensor([-1.0, -2.0, -3.0], device='cuda')
    results["test_case_2"] = erfc_sqrt(input2)

    # Test case 3: Mixed values
    input3 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_3"] = erfc_sqrt(input3)

    # Test case 4: Zero values
    input4 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_4"] = erfc_sqrt(input4)

    return results

test_results = test_erfc_sqrt()
