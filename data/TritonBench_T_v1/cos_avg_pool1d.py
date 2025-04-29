import torch
import torch.nn.functional as F

def cos_avg_pool1d(
        input: torch.Tensor, 
        kernel_size: int, 
        stride: int=None, 
        padding: int=0, 
        ceil_mode: bool=False, 
        count_include_pad: bool=True) -> torch.Tensor:
    """
    Applies the cosine function element-wise to the input tensor, followed by 1D average pooling.

    Args:
        input (Tensor): The input tensor of shape (minibatch, in_channels, iW).
        kernel_size (int): Size of the pooling window.
        stride (int, optional): Stride of the pooling window. Defaults to `kernel_size`.
        padding (int, optional): Zero-padding added to both sides of the input. Default is 0.
        ceil_mode (bool, optional): If True, uses ceil instead of floor to compute the output shape. Default is False.
        count_include_pad (bool, optional): If True, includes the zero-padding in the averaging calculation. Default is True.

    Returns:
        Tensor: The resulting tensor after cosine transformation and 1D average pooling.
    """
    cos_input = torch.cos(input)
    return F.avg_pool1d(cos_input, kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad)

##################################################################################################################################################


import torch
torch.manual_seed(42)


def test_cos_avg_pool1d():
    results = {}

    # Test case 1: Basic functionality with default parameters
    input_tensor_1 = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0]]], device='cuda')
    results['test_case_1'] = cos_avg_pool1d(input_tensor_1, kernel_size=2)

    # Test case 2: Custom stride
    input_tensor_2 = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0]]], device='cuda')
    results['test_case_2'] = cos_avg_pool1d(input_tensor_2, kernel_size=2, stride=1)

    # Test case 3: With padding
    input_tensor_3 = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0]]], device='cuda')
    results['test_case_3'] = cos_avg_pool1d(input_tensor_3, kernel_size=2, padding=1)

    # Test case 4: Using ceil_mode
    input_tensor_4 = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0]]], device='cuda')
    results['test_case_4'] = cos_avg_pool1d(input_tensor_4, kernel_size=2, ceil_mode=True)

    return results

test_results = test_cos_avg_pool1d()
print(test_results)