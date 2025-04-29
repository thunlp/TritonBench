import torch
import torch.nn.functional as F

def grid_sample_with_affine(
        input: torch.Tensor, 
        theta: torch.Tensor, 
        size: torch.Size, 
        mode: str='bilinear', 
        padding_mode: str='zeros', 
        align_corners: bool=False) -> torch.Tensor:
    """
    Apply an affine transformation followed by grid sampling to the input tensor.

    Parameters:
    - input (torch.Tensor): Input tensor of shape (N, C, H_in, W_in)
    - theta (torch.Tensor): Affine transformation matrix of shape (N, 2, 3)
    - size (torch.Size): Target output image size (N, C, H_out, W_out)
    - mode (str): Interpolation mode for grid sampling ('bilinear', 'nearest', or 'bicubic'). Default is 'bilinear'.
    - padding_mode (str): Defines how to handle grid values outside the input range ('zeros', 'border', 'reflection'). Default is 'zeros'.
    - align_corners (bool): If True, aligns the grid to corner pixels for transformation consistency. Default is False.

    Returns:
    - torch.Tensor: Output tensor of shape (N, C, H_out, W_out) after affine transformation and grid sampling.
    """
    # Ensure theta has a floating point type
    theta = theta.float()
    
    # Create a grid for affine transformation
    grid = F.affine_grid(theta, size, align_corners=align_corners)
    
    # Perform grid sampling
    output = F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    
    return output

##################################################################################################################################################

import torch
torch.manual_seed(42)

def test_grid_sample_with_affine():
    results = {}

    # Test Case 1: Default parameters
    input_tensor = torch.randn(1, 3, 64, 64, device='cuda')
    theta = torch.tensor([[[1, 0, 0], [0, 1, 0]]], device='cuda')  # Affine matrix as int64
    size = torch.Size((1, 3, 64, 64))
    results["test_case_1"] = grid_sample_with_affine(input_tensor, theta, size)

    # Test Case 2: Nearest mode
    results["test_case_2"] = grid_sample_with_affine(input_tensor, theta, size, mode='nearest')

    # Test Case 3: Reflection padding mode
    results["test_case_3"] = grid_sample_with_affine(input_tensor, theta, size, padding_mode='reflection')

    # Test Case 4: Align corners
    results["test_case_4"] = grid_sample_with_affine(input_tensor, theta, size, align_corners=True)

    return results

test_results = test_grid_sample_with_affine()print(test_results)