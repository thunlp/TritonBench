import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Union, Tuple
import torch

def gelu_conv2d(input: Tensor, weight: Tensor, bias: Optional[Tensor]=None, stride: Union[int, Tuple[int, int]]=1, padding: Union[int, Tuple[int, int], str]=0, dilation: Union[int, Tuple[int, int]]=1, groups: int=1, approximate: str='none', out: Optional[Tensor]=None) -> Tensor:
    conv_result = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    return F.gelu(conv_result, approximate=approximate, out=out)

##################################################################################################################################################


import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Union, Tuple

# def gelu_conv2d(input: Tensor, weight: Tensor, bias: Optional[Tensor]=None, stride: Union[int, Tuple[int, int]]=1, padding: Union[int, Tuple[int, int], str]=0, dilation: Union[int, Tuple[int, int]]=1, groups: int=1, approximate: str='none', out: Optional[Tensor]=None) -> Tensor:
#     conv_result = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
#     return F.gelu(conv_result, approximate=approximate, out=out)

def test_gelu_conv2d():
    results = {}

    # Test case 1: Basic test with default parameters
    input1 = torch.randn(1, 3, 5, 5, device='cuda')
    weight1 = torch.randn(2, 3, 3, 3, device='cuda')
    results["test_case_1"] = gelu_conv2d(input1, weight1)

    # Test case 2: Test with bias
    input2 = torch.randn(1, 3, 5, 5, device='cuda')
    weight2 = torch.randn(2, 3, 3, 3, device='cuda')
    bias2 = torch.randn(2, device='cuda')
    results["test_case_2"] = gelu_conv2d(input2, weight2, bias=bias2)

    # Test case 3: Test with stride and padding
    input3 = torch.randn(1, 3, 8, 8, device='cuda')
    weight3 = torch.randn(2, 3, 3, 3, device='cuda')
    results["test_case_3"] = gelu_conv2d(input3, weight3, stride=2, padding=1)

    # Test case 4: Test with dilation and groups
    input4 = torch.randn(1, 4, 10, 10, device='cuda')
    weight4 = torch.randn(4, 1, 3, 3, device='cuda')
    results["test_case_4"] = gelu_conv2d(input4, weight4, dilation=2, groups=4)

    return results

test_results = test_gelu_conv2d()
