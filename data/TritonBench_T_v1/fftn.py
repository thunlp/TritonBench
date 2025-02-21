import torch

def fftn(input, s=None, dim=None, norm=None, out=None):
    return torch.fft.fftn(input, s=s, dim=dim, norm=norm)

##################################################################################################################################################


import torch

def fftn(input, s=None, dim=None, norm=None, out=None):
    return torch.fft.fftn(input, s=s, dim=dim, norm=norm)

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
