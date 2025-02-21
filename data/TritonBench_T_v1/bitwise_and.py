import torch

def bitwise_and(input, other, out=None):
    """
    Computes the bitwise AND of two tensors. The input tensors must be of integral or boolean types.
    For boolean tensors, it computes the logical AND.

    Args:
        input (Tensor): The first input tensor, should be of integral or boolean type.
        other (Tensor): The second input tensor, should be of integral or boolean type.
        out (Tensor, optional): The output tensor where the result will be stored. Defaults to None.

    Returns:
        Tensor: A tensor containing the result of the bitwise AND operation.
    """
    return torch.bitwise_and(input, other, out=out)

##################################################################################################################################################


import torch

def test_bitwise_and():
    results = {}

    # Test case 1: Bitwise AND with integer tensors
    input1 = torch.tensor([1, 2, 3], dtype=torch.int32, device='cuda')
    other1 = torch.tensor([3, 2, 1], dtype=torch.int32, device='cuda')
    results["test_case_1"] = bitwise_and(input1, other1)

    # Test case 2: Bitwise AND with boolean tensors
    input2 = torch.tensor([True, False, True], dtype=torch.bool, device='cuda')
    other2 = torch.tensor([False, False, True], dtype=torch.bool, device='cuda')
    results["test_case_2"] = bitwise_and(input2, other2)

    # Test case 3: Bitwise AND with different shapes (broadcasting)
    input3 = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32, device='cuda')
    other3 = torch.tensor([1, 0], dtype=torch.int32, device='cuda')
    results["test_case_3"] = bitwise_and(input3, other3)

    # Test case 4: Bitwise AND with scalar tensor
    input4 = torch.tensor([1, 2, 3], dtype=torch.int32, device='cuda')
    other4 = torch.tensor(2, dtype=torch.int32, device='cuda')
    results["test_case_4"] = bitwise_and(input4, other4)

    return results

test_results = test_bitwise_and()
