import torch
import torch.nn.functional as F


def grid_sample(
        input: torch.Tensor, 
        grid: torch.Tensor, 
        mode: str='bilinear', 
        padding_mode: str='zeros', 
        align_corners: bool=False) -> torch.Tensor:
    """
    Performs grid sampling using the specified input and grid.

    Parameters:
    - input (Tensor): The input tensor (4D or 5D). For 4D: (N, C, H, W), for 5D: (N, C, D, H, W).
    - grid (Tensor): The grid tensor, which provides the sampling points. Should be in the range [-1, 1].
    - mode (str, optional): The interpolation mode. Can be 'bilinear' (default) or 'nearest'.
    - padding_mode (str, optional): Defines the padding mode when grid values are outside the valid range. Can be 'zeros', 'border', or 'reflection'.
    - align_corners (bool, optional): If True, the corners of the grid will align with the corners of the input.

    Returns:
    - Tensor: The output tensor after performing grid sampling.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError('Input should be a torch.Tensor.')
    if not isinstance(grid, torch.Tensor):
        raise TypeError('Grid should be a torch.Tensor.')
    if mode not in ['bilinear', 'nearest']:
        raise ValueError("Mode should be either 'bilinear' or 'nearest'.")
    if padding_mode not in ['zeros', 'border', 'reflection']:
        raise ValueError("Padding mode should be one of 'zeros', 'border', or 'reflection'.")
    output = F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    return output

##################################################################################################################################################


import torch

def test_grid_sample():
    results = {}

    # Test case 1: 4D input, bilinear mode, zeros padding
    input_4d = torch.rand(1, 3, 4, 4, device='cuda')
    grid_4d = torch.rand(1, 2, 2, 2, device='cuda') * 2 - 1  # Range [-1, 1]
    results["test_case_1"] = grid_sample(input_4d, grid_4d)

    # Test case 2: 4D input, nearest mode, border padding
    results["test_case_2"] = grid_sample(input_4d, grid_4d, mode='nearest', padding_mode='border')

    # Test case 3: 5D input, bilinear mode, reflection padding
    input_5d = torch.rand(1, 3, 4, 4, 4, device='cuda')
    grid_5d = torch.rand(1, 2, 2, 2, 3, device='cuda') * 2 - 1  # Range [-1, 1]
    results["test_case_3"] = grid_sample(input_5d, grid_5d, padding_mode='reflection')

    # Test case 4: 5D input, nearest mode, zeros padding, align_corners=True
    results["test_case_4"] = grid_sample(input_5d, grid_5d, mode='nearest', align_corners=True)

    return results

test_results = test_grid_sample()
print(test_results)