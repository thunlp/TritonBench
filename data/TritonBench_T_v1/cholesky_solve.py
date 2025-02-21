import torch

def cholesky_solve(B, L, upper=False, out=None):
    """
    计算给定Cholesky分解的对称正定矩阵的线性方程组的解。
    
    参数：
    B (Tensor): 右侧张量，形状为(*, n, k)，其中*是零个或多个批次维度。
    L (Tensor): 形状为(*, n, n)的张量，表示对称或厄米正定矩阵的Cholesky分解，包含下三角或上三角。
    upper (bool, optional): 标志，指示L是否是上三角。默认值为False（表示L是下三角）。
    out (Tensor, optional): 输出张量。如果为None，则返回一个新的张量。
    
    返回：
    Tensor: 解矩阵X，形状与B相同。
    """
    return torch.cholesky_solve(B, L, upper=upper, out=out)

##################################################################################################################################################


import torch

def test_cholesky_solve():
    results = {}

    # Test case 1: Lower triangular matrix
    B1 = torch.tensor([[1.0], [2.0]], device='cuda')
    L1 = torch.tensor([[2.0, 0.0], [1.0, 1.0]], device='cuda')
    results["test_case_1"] = cholesky_solve(B1, L1)

    # Test case 2: Upper triangular matrix
    B2 = torch.tensor([[1.0], [2.0]], device='cuda')
    L2 = torch.tensor([[2.0, 1.0], [0.0, 1.0]], device='cuda')
    results["test_case_2"] = cholesky_solve(B2, L2, upper=True)

    # Test case 3: Batch of matrices, lower triangular
    B3 = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]], device='cuda')
    L3 = torch.tensor([[[2.0, 0.0], [1.0, 1.0]], [[3.0, 0.0], [1.0, 2.0]]], device='cuda')
    results["test_case_3"] = cholesky_solve(B3, L3)

    # Test case 4: Batch of matrices, upper triangular
    B4 = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]], device='cuda')
    L4 = torch.tensor([[[2.0, 1.0], [0.0, 1.0]], [[3.0, 1.0], [0.0, 2.0]]], device='cuda')
    results["test_case_4"] = cholesky_solve(B4, L4, upper=True)

    return results

test_results = test_cholesky_solve()
