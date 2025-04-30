import torch
from typing import Optional

def sigmoid_argmax(input: torch.Tensor, dim: Optional[int]=None, keepdim: bool=False) -> torch.Tensor:
    """
    Apply sigmoid to each element of the input tensor, then return the indices of the maximum values
    along the specified dimension or over all elements if no dimension is specified.
    
    Parameters:
    - input (torch.Tensor): The input tensor.
    - dim (int, optional): The dimension to reduce. Default is None, which computes the argmax over all elements.
    - keepdim (bool, optional): Whether the output tensor has :attr:`dim` retained or not. Default is False.
    
    Returns:
    - torch.Tensor: The indices of the maximum values.
    """
    sigmoid_output = torch.sigmoid(input)
    return torch.argmax(sigmoid_output, dim=dim, keepdim=keepdim)

##################################################################################################################################################


import torch
torch.manual_seed(42)
def test_sigmoid_argmax():
    results = {}

    # Test case 1: 1D tensor, no dim specified
    input1 = torch.tensor([0.1, 2.0, -1.0, 3.0], device='cuda')
    results["test_case_1"] = sigmoid_argmax(input1)

    # Test case 2: 2D tensor, dim=0
    input2 = torch.tensor([[0.1, 2.0, -1.0], [3.0, -0.5, 1.5]], device='cuda')
    results["test_case_2"] = sigmoid_argmax(input2, dim=0)

    # Test case 3: 2D tensor, dim=1
    input3 = torch.tensor([[0.1, 2.0, -1.0], [3.0, -0.5, 1.5]], device='cuda')
    results["test_case_3"] = sigmoid_argmax(input3, dim=1)

    # Test case 4: 2D tensor, dim=1, keepdim=True
    input4 = torch.tensor([[0.1, 2.0, -1.0], [3.0, -0.5, 1.5]], device='cuda')
    results["test_case_4"] = sigmoid_argmax(input4, dim=1, keepdim=True)

    return results

test_results = test_sigmoid_argmax()
print(test_results)