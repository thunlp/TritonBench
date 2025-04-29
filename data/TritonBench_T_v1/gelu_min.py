import torch
from typing import Optional, Union, Tuple
def gelu_min(
        input: torch.Tensor, 
        approximate: str='none', 
        dim: Optional[int]=None, 
        keepdim: bool=False, 
        out: Optional[torch.Tensor]=None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Applies the GELU activation function followed by a minimum operation.

    Args:
        input (torch.Tensor): The input tensor.
        approximate (str, optional): The approximation method for GELU. Default is 'none'.
        dim (int, optional): The dimension to reduce. Default is None.
        keepdim (bool, optional): Whether to keep the reduced dimension. Default is False.
        out (torch.Tensor, optional): The output tensor. Default is None.   

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: The result of the fused operation.
    """
    
    if approximate == 'none':
        output = input * torch.erf(input / torch.sqrt(torch.tensor(2.0))) / 2.0
    elif approximate == 'tanh':
        output = 0.5 * input * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (input + 0.044715 * input ** 3)))
    else:
        raise ValueError("Unknown approximation method. Choose either 'none' or 'tanh'.")
    if dim is None:
        return torch.min(output)
    else:
        (min_values, indices) = torch.min(output, dim=dim, keepdim=keepdim)
        if out is not None:
            out[0].copy_(min_values)
            out[1].copy_(indices)
        return (min_values, indices)

##################################################################################################################################################


def test_gelu_min():
    results = {}

    # Test case 1: Default approximate='none', no dim, no keepdim
    input_tensor = torch.tensor([0.5, -0.5, 1.0, -1.0], device='cuda')
    results['test_case_1'] = gelu_min(input_tensor)

    # Test case 2: approximate='tanh', no dim, no keepdim
    input_tensor = torch.tensor([0.5, -0.5, 1.0, -1.0], device='cuda')
    results['test_case_2'] = gelu_min(input_tensor, approximate='tanh')

    # Test case 3: approximate='none', with dim, no keepdim
    input_tensor = torch.tensor([[0.5, -0.5], [1.0, -1.0]], device='cuda')
    results['test_case_3'] = gelu_min(input_tensor, dim=1)

    # Test case 4: approximate='tanh', with dim, keepdim=True
    input_tensor = torch.tensor([[0.5, -0.5], [1.0, -1.0]], device='cuda')
    results['test_case_4'] = gelu_min(input_tensor, approximate='tanh', dim=1, keepdim=True)

    return results

test_results = test_gelu_min()
print(test_results)