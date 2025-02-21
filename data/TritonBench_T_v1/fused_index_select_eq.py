import torch

def fused_index_select_eq(input, dim, index, other, *, out=None):
    """
    Perform a fused operation combining index selection and element-wise equality comparison.

    Args:
        input (Tensor): The input tensor X.
        dim (int): The dimension along which to index.
        index (IntTensor or LongTensor): The indices to select along dimension dim.
        other (Tensor or float): The tensor or value Y to compare with the selected tensor.
        out (Tensor, optional): Output tensor. Ignored if None. Default: None.

    Returns:
        Tensor: A boolean tensor of the same shape as the selected elements, indicating where the comparisons are true.
    """
    selected = torch.index_select(input, dim, index)
    output = torch.eq(selected, other)
    if out is not None:
        out.copy_(output)
        return out
    return output

##################################################################################################################################################


import torch

def test_fused_index_select_eq():
    results = {}

    # Test case 1: Basic functionality
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
    dim = 0
    index = torch.tensor([0, 1], device='cuda')
    other = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
    results["test_case_1"] = fused_index_select_eq(input_tensor, dim, index, other)

    # Test case 2: Different dimension
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
    dim = 1
    index = torch.tensor([0, 2], device='cuda')
    other = torch.tensor([[1, 3], [4, 6]], device='cuda')
    results["test_case_2"] = fused_index_select_eq(input_tensor, dim, index, other)

    # Test case 3: Scalar comparison
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
    dim = 1
    index = torch.tensor([1], device='cuda')
    other = 2
    results["test_case_3"] = fused_index_select_eq(input_tensor, dim, index, other)

    # Test case 4: No output tensor provided
    input_tensor = torch.tensor([[7, 8, 9], [10, 11, 12]], device='cuda')
    dim = 0
    index = torch.tensor([1], device='cuda')
    other = torch.tensor([[10, 11, 12]], device='cuda')
    results["test_case_4"] = fused_index_select_eq(input_tensor, dim, index, other)

    return results

test_results = test_fused_index_select_eq()
