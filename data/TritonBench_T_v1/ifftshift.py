import torch

def ifftshift(input, dim=None):
    """
    Perform the inverse FFT shift on the input tensor.

    Args:
        input (Tensor): the tensor in FFT order.
        dim (int, Tuple[int], optional): The dimensions to rearrange.
            Only dimensions specified here will be rearranged,
            any other dimensions will be left in their original order.
            Default: All dimensions of input.

    Returns:
        Tensor: the tensor after inverse FFT shift.
    """
    return torch.fft.ifftshift(input, dim)

##################################################################################################################################################


import torch

def test_ifftshift():
    results = {}

    # Test case 1: 1D tensor, default dim
    input_tensor_1d = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device='cuda')
    results["test_case_1"] = ifftshift(input_tensor_1d)

    # Test case 2: 2D tensor, default dim
    input_tensor_2d = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]], device='cuda')
    results["test_case_2"] = ifftshift(input_tensor_2d)

    # Test case 3: 2D tensor, specific dim
    results["test_case_3"] = ifftshift(input_tensor_2d, dim=0)

    # Test case 4: 3D tensor, specific dim
    input_tensor_3d = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], device='cuda')
    results["test_case_4"] = ifftshift(input_tensor_3d, dim=(1, 2))

    return results

test_results = test_ifftshift()
