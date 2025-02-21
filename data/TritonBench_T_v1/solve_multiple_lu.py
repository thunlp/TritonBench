import torch

def solve_multiple_lu(A, Bs, *, pivot=True, out=None):
    (P, L, U) = torch.linalg.lu(A, pivot=pivot)
    if pivot:
        Bs_perm = torch.matmul(P.transpose(-2, -1), Bs)
    else:
        Bs_perm = Bs
    Y = torch.linalg.solve_triangular(L, Bs_perm, upper=False, unitriangular=True)
    X = torch.linalg.solve_triangular(U, Y, upper=True)
    if out is not None:
        out.copy_(X)
        return out
    return X

##################################################################################################################################################


import torch

def test_solve_multiple_lu():
    results = {}

    # Test case 1: Basic test with pivot=True
    A1 = torch.tensor([[3.0, 1.0], [1.0, 2.0]], device='cuda')
    Bs1 = torch.tensor([[9.0], [8.0]], device='cuda')
    results["test_case_1"] = solve_multiple_lu(A1, Bs1)

    # Test case 2: Test with pivot=False
    A2 = torch.tensor([[4.0, 3.0], [6.0, 3.0]], device='cuda')
    Bs2 = torch.tensor([[10.0], [12.0]], device='cuda')
    results["test_case_2"] = solve_multiple_lu(A2, Bs2, pivot=False)

    # Test case 3: Test with a batch of Bs
    A3 = torch.tensor([[2.0, 0.0], [0.0, 2.0]], device='cuda')
    Bs3 = torch.tensor([[4.0, 6.0], [8.0, 10.0]], device='cuda')
    results["test_case_3"] = solve_multiple_lu(A3, Bs3)

    # Test case 4: Test with a larger matrix
    A4 = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]], device='cuda')
    Bs4 = torch.tensor([[14.0], [10.0], [18.0]], device='cuda')
    results["test_case_4"] = solve_multiple_lu(A4, Bs4)

    return results

test_results = test_solve_multiple_lu()
