import torch
from typing import Optional
def rand(*size: int, 
        generator: Optional[torch.Generator]=None, 
        out: Optional[torch.Tensor]=None, 
        dtype: Optional[torch.dtype]=None, 
        layout: torch.layout=torch.strided, 
        device: Optional[torch.device]=None, 
        requires_grad: bool=False, 
        pin_memory: bool=False) -> torch.Tensor:
    """
    Generates a tensor with random numbers from a uniform distribution on the interval [0, 1).

    Args:
        size (int...): A sequence of integers defining the shape of the output tensor.
        generator (torch.Generator, optional): A pseudorandom number generator for sampling.
        out (torch.Tensor, optional): The output tensor.
        dtype (torch.dtype, optional): The desired data type of returned tensor.
        layout (torch.layout, optional): The desired layout of returned Tensor.
        device (torch.device, optional): The desired device of returned tensor.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor.
        pin_memory (bool, optional): If set, returned tensor would be allocated in the pinned memory (CPU only).

    Returns:
        torch.Tensor: A tensor of shape `size` with random numbers in the interval [0, 1).
    """
    return torch.rand(*size, generator=generator, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, pin_memory=pin_memory)

##################################################################################################################################################


import torch

def test_rand():
    results = {}

    # Test case 1: Basic usage with size only
    results["test_case_1"] = rand(2, 3, device='cuda')

    # Test case 2: Specifying dtype
    results["test_case_2"] = rand(2, 3, dtype=torch.float64, device='cuda')

    # Test case 3: Using a generator
    gen = torch.Generator(device='cuda')
    gen.manual_seed(42)
    results["test_case_3"] = rand(2, 3, generator=gen, device='cuda')

    # Test case 4: Requires gradient
    results["test_case_4"] = rand(2, 3, requires_grad=True, device='cuda')

    return results

test_results = test_rand()
print(test_results)