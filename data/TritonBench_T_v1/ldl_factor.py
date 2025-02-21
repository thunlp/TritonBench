import torch

def ldl_factor(A, hermitian=False, out=None):
    """
    Perform the LDL factorization of a symmetric or Hermitian matrix.

    Args:
        A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions consisting of symmetric or Hermitian matrices.
        hermitian (bool, optional): whether to consider the input to be Hermitian or symmetric. Default is False.
        out (tuple, optional): tuple of two tensors to write the output to. Ignored if None. Default is None.

    Returns:
        namedtuple: A named tuple `(LD, pivots)`. 
                    LD is the compact representation of L and D.
                    pivots is a tensor containing the pivot indices.
    """
    (LD, pivots) = torch.linalg.ldl_factor(A, hermitian=hermitian, out=out)
    return (LD, pivots)

##################################################################################################################################################


import torch

def test_ldl_factor():
    results = {}

    # Test case 1: Symmetric matrix
    A1 = torch.tensor([[4.0, 1.0], [1.0, 3.0]], device='cuda')
    results["test_case_1"] = ldl_factor(A1)

    # Test case 2: Hermitian matrix
    A2 = torch.tensor([[2.0, 1.0j], [-1.0j, 2.0]], device='cuda')
    results["test_case_2"] = ldl_factor(A2, hermitian=True)

    # Test case 3: Batch of symmetric matrices
    A3 = torch.tensor([[[4.0, 1.0], [1.0, 3.0]], [[2.0, 0.5], [0.5, 2.0]]], device='cuda')
    results["test_case_3"] = ldl_factor(A3)

    # Test case 4: Batch of Hermitian matrices
    A4 = torch.tensor([[[2.0, 1.0j], [-1.0j, 2.0]], [[3.0, 0.5j], [-0.5j, 3.0]]], device='cuda')
    results["test_case_4"] = ldl_factor(A4, hermitian=True)

    return results

test_results = test_ldl_factor()
