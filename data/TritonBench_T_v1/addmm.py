import torch

def addmm(input: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor, beta: float=1, alpha: float=1, out: torch.Tensor=None) -> torch.Tensor:
    """
    Performs matrix multiplication of mat1 and mat2, and adds input to the result.

    Parameters:
        input (torch.Tensor): Matrix to be added.
        mat1 (torch.Tensor): The first matrix to be matrix-multiplied.
        mat2 (torch.Tensor): The second matrix to be matrix-multiplied.
        beta (float, optional): Multiplier for input (default is 1).
        alpha (float, optional): Multiplier for mat1 @ mat2 (default is 1).
        out (torch.Tensor, optional): The output tensor to store the result.

    Returns:
        torch.Tensor: The resulting tensor after performing the operation.
    
    This function performs the matrix multiplication of mat1 and mat2, scales the result by alpha,
    and then adds it to the input matrix scaled by beta. The resulting matrix is returned.
    
    If input is sparse, the result will have the same layout as input. If out is provided,
    it must have the same layout as input. If beta is 0, the input will be ignored, and nan or inf
    in input will not be propagated.
    """
    return torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha, out=out)

##################################################################################################################################################


import torch

def test_addmm():
    results = {}

    # Test case 1: Default beta and alpha
    input1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    mat1_1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    mat2_1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    results["test_case_1"] = addmm(input1, mat1_1, mat2_1)

    # Test case 2: Custom beta and alpha
    input2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    mat1_2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    mat2_2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    results["test_case_2"] = addmm(input2, mat1_2, mat2_2, beta=0.5, alpha=2.0)

    # Test case 3: Zero beta
    input3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    mat1_3 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    mat2_3 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    results["test_case_3"] = addmm(input3, mat1_3, mat2_3, beta=0.0)

    return results

test_results = test_addmm()
