import torch
from typing import Union, List, Tuple

def tensordot(a: torch.Tensor, b: torch.Tensor, dims: Union[int, Tuple[List[int], List[int]], List[List[int]]]) -> torch.Tensor:
    """
    Perform a generalized matrix product between tensors a and b along specified dimensions.
    
    Args:
        a (torch.Tensor): Left tensor to contract
        b (torch.Tensor): Right tensor to contract
        dims (Union[int, Tuple[List[int], List[int]], List[List[int]]]): 
            - If int, the number of dimensions to contract (e.g., dims=2 means contraction over the last 2 dimensions of a and the first 2 dimensions of b).
            - If tuple, should contain two lists of dimensions to contract over for tensors a and b, respectively.
            - If list of lists, it can define multiple contraction axes between tensors.
    
    Returns:
        torch.Tensor: The result of the contraction between tensors a and b over the specified dimensions.
    """
    return torch.tensordot(a, b, dims)

##################################################################################################################################################


import torch
from typing import Union, List, Tuple

def test_tensordot():
    results = {}
    
    # 示例 1
    a = torch.arange(60.).reshape(3, 4, 5)
    b = torch.arange(24.).reshape(4, 3, 2)
    results["test_case_1"] = tensordot(a, b, dims=([1, 0], [0, 1]))

    # 示例 2 (在CUDA设备上)
    a = torch.randn(3, 4, 5, device='cuda')
    b = torch.randn(4, 5, 6, device='cuda')
    results["test_case_2"] = tensordot(a, b, dims=2).cpu()

    # 示例 3 (多维收缩)
    a = torch.randn(3, 5, 4, 6)
    b = torch.randn(6, 4, 5, 3)
    results["test_case_3"] = tensordot(a, b, dims=([2, 1, 3], [1, 2, 0]))
    
    return results

test_results = test_tensordot()
