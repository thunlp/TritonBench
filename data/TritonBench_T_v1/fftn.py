import torch

def fftn(input: torch.Tensor, s=None, dim=None, norm=None, out=None) -> torch.Tensor:
    """
    Computes the N-dimensional discrete Fourier Transform of the input tensor.

    Args:
        input (torch.Tensor): The input tensor to compute the N-dimensional Fourier Transform on.
        s (tuple, optional): The size of the output tensor.
        dim (tuple, optional): The dimensions to transform.

    Returns:
        torch.Tensor: The N-dimensional discrete Fourier Transform of the input tensor.
    """
    fftn = torch.fft.fftn(input, s=s, dim=dim, norm=norm)
    if out is not None:
        out.copy_(fftn)
        return out
    return fftn

##################################################################################################################################################


import torch
torch.manual_seed(42)


def test_fftn():
    results = {}
    
    # Test case 1: Only input tensor
    input_tensor = torch.randn(4, 4, device='cuda')
    results["test_case_1"] = fftn(input_tensor)
    
    # Test case 2: Input tensor with s parameter
    input_tensor = torch.randn(4, 4, device='cuda')
    s = (2, 2)
    results["test_case_2"] = fftn(input_tensor, s=s)
    
    # Test case 3: Input tensor with dim parameter
    input_tensor = torch.randn(4, 4, device='cuda')
    dim = (0, 1)
    results["test_case_3"] = fftn(input_tensor, dim=dim)
    
    # Test case 4: Input tensor with norm parameter
    input_tensor = torch.randn(4, 4, device='cuda')
    norm = "ortho"
    results["test_case_4"] = fftn(input_tensor, norm=norm)
    
    return results

test_results = test_fftn()
print(test_results)