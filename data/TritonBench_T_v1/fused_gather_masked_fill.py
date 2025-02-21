import torch

def fused_gather_masked_fill(input, dim, index, mask, value, *, sparse_grad=False, out=None):
    """
    Combines torch.gather and torch.Tensor.masked_fill into a single operation.
    
    Arguments:
    input (Tensor) -- the input tensor X.
    dim (int) -- the dimension along which to index.
    index (LongTensor) -- the indices of elements to gather, same dimensionality as `input`.
    mask (BoolTensor) -- a boolean mask tensor, broadcastable to the shape of the output tensor.
    value (float) -- the value to fill where `mask` is True.
    sparse_grad (bool, optional) -- If True, gradient w.r.t. `input` will be sparse. Default: False.
    out (Tensor, optional) -- output tensor. If None, a new tensor will be returned. Default: None.
    
    Returns:
    Tensor -- the resulting tensor after gather and masked fill operations.
    """
    gathered = torch.gather(input, dim, index, sparse_grad=sparse_grad)
    output = gathered.masked_fill(mask, value)
    if out is not None:
        out.copy_(output)
        return out
    return output

##################################################################################################################################################


import torch

def test_fused_gather_masked_fill():
    results = {}

    # Test case 1: Basic functionality
    input1 = torch.tensor([[1, 2], [3, 4]], device='cuda')
    index1 = torch.tensor([[0, 1], [1, 0]], device='cuda')
    mask1 = torch.tensor([[True, False], [False, True]], device='cuda')
    value1 = -1.0
    results["test_case_1"] = fused_gather_masked_fill(input1, 1, index1, mask1, value1)

    # Test case 2: Different dimension
    input2 = torch.tensor([[5, 6, 7], [8, 9, 10]], device='cuda')
    index2 = torch.tensor([[0, 2], [1, 0]], device='cuda')
    mask2 = torch.tensor([[False, True], [True, False]], device='cuda')
    value2 = 0.0
    results["test_case_2"] = fused_gather_masked_fill(input2, 1, index2, mask2, value2)

    # Test case 3: Sparse gradient
    input3 = torch.tensor([[11, 12], [13, 14]], device='cuda')
    index3 = torch.tensor([[1, 0], [0, 1]], device='cuda')
    mask3 = torch.tensor([[True, True], [False, False]], device='cuda')
    value3 = 99.0
    results["test_case_3"] = fused_gather_masked_fill(input3, 1, index3, mask3, value3, sparse_grad=True)

    # Test case 4: Larger tensor
    input4 = torch.tensor([[15, 16, 17, 18], [19, 20, 21, 22]], device='cuda')
    index4 = torch.tensor([[3, 2, 1, 0], [0, 1, 2, 3]], device='cuda')
    mask4 = torch.tensor([[False, False, True, True], [True, False, False, True]], device='cuda')
    value4 = -5.0
    results["test_case_4"] = fused_gather_masked_fill(input4, 1, index4, mask4, value4)

    return results

test_results = test_fused_gather_masked_fill()
