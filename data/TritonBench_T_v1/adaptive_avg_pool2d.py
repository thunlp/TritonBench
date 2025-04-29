import torch
import torch.nn.functional as F
from typing import Union, Tuple
def adaptive_avg_pool2d(input: torch.Tensor, output_size: Union[int, Tuple[int, int]]) -> torch.Tensor:
    """
    Apply 2D adaptive average pooling over an input signal.

    Args:
        input (Tensor): The input tensor, either of shape (N, C, H_in, W_in) or (C, H_in, W_in).
        output_size (int or tuple): The target output size (single integer or tuple of two integers).
        
            - If an integer, the output will be square: (output_size, output_size).
            - If a tuple, the first element corresponds to the height, and the second element corresponds to the width of the output.

    Returns:
        Tensor: The output tensor with the specified output size.

    Example:
        >>> import torch
        >>> from adaptive_avg_pool2d import adaptive_avg_pool2d
        >>> input = torch.randn(1, 64, 8, 9)
        >>> output = adaptive_avg_pool2d(input, (5, 7))
        >>> print(output.shape)  # Output shape: (1, 64, 5, 7)
    """
    return F.adaptive_avg_pool2d(input, output_size)

##################################################################################################################################################


import torch
torch.manual_seed(42)

def test_adaptive_avg_pool2d():
    results = {}
    
    # Test case 1: input shape (N, C, H_in, W_in), output_size as integer
    input1 = torch.randn(1, 64, 8, 9).cuda()
    output1 = adaptive_avg_pool2d(input1, 5)
    results["test_case_1"] = output1
    
    # Test case 2: input shape (N, C, H_in, W_in), output_size as tuple
    input2 = torch.randn(1, 64, 8, 9).cuda()
    output2 = adaptive_avg_pool2d(input2, (5, 7))
    results["test_case_2"] = output2
    
    # Test case 3: input shape (C, H_in, W_in), output_size as integer
    input3 = torch.randn(64, 8, 9).cuda()
    output3 = adaptive_avg_pool2d(input3, 5)
    results["test_case_3"] = output3
    
    # Test case 4: input shape (C, H_in, W_in), output_size as tuple
    input4 = torch.randn(64, 8, 9).cuda()
    output4 = adaptive_avg_pool2d(input4, (5, 7))
    results["test_case_4"] = output4
    
    return results

test_results = test_adaptive_avg_pool2d()
print(test_results)