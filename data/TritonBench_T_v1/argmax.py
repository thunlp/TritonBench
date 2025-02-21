import torch

def argmax(input_tensor, dim, keepdim=False):
    """
    Returns the indices of the maximum values of a tensor across a specified dimension.

    Args:
        input_tensor (Tensor): The input tensor.
        dim (int): The dimension to reduce. If None, the argmax of the flattened input is returned.
        keepdim (bool): Whether the output tensor has the dimension retained or not.

    Returns:
        LongTensor: A tensor containing the indices of the maximum values.
    """
    return torch.argmax(input_tensor, dim=dim, keepdim=keepdim)

##################################################################################################################################################


import torch

def test_argmax():
    results = {}

    # Test case 1: 2D tensor, dim=0
    tensor_2d = torch.tensor([[1, 3, 2], [4, 0, 5]], device='cuda')
    results["test_case_1"] = argmax(tensor_2d, dim=0)

    # Test case 2: 2D tensor, dim=1
    results["test_case_2"] = argmax(tensor_2d, dim=1)

    # Test case 3: 3D tensor, dim=2
    tensor_3d = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], device='cuda')
    results["test_case_3"] = argmax(tensor_3d, dim=2)

    # Test case 4: 3D tensor, dim=1, keepdim=True
    results["test_case_4"] = argmax(tensor_3d, dim=1, keepdim=True)

    return results

test_results = test_argmax()
