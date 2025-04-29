import torch

def SGD_step(parameters: torch.Tensor, grads: torch.Tensor, lr=0.1):
    """
    Performs a single step of SGD optimization.
    Updates parameters in-place.
    """
    with torch.no_grad():
        for param, grad in zip(parameters, grads):
            if grad is None:
                continue
            # Update rule: param = param - lr * grad
            param.add_(grad, alpha=-lr)

##################################################################################################################################################

def test_SGD():
    results = {}

    # Test case 1: Basic functionality
    params1 = [torch.ones(2, 2, requires_grad=True, device='cuda'), torch.zeros(3, requires_grad=True, device='cuda')]
    grads1 = [torch.full_like(params1[0], 2.0), torch.full_like(params1[1], -1.0)]
    expected_params1 = [params1[0].clone() - 0.1 * grads1[0], params1[1].clone() - 0.1 * grads1[1]]
    SGD_step(params1, grads1, lr=0.1)
    results["test_case_1_param0"] = params1[0]
    results["test_case_1_param1"] = params1[1]


    # Test case 2: Different learning rate
    params2 = [torch.ones(2, 2, requires_grad=True, device='cuda'), torch.zeros(3, requires_grad=True, device='cuda')]
    grads2 = [torch.full_like(params2[0], 2.0), torch.full_like(params2[1], -1.0)]
    lr2 = 0.01
    expected_params2 = [params2[0].clone() - lr2 * grads2[0], params2[1].clone() - lr2 * grads2[1]]
    SGD_step(params2, grads2, lr=lr2)
    results["test_case_2_param0"] = params2[0]
    results["test_case_2_param1"] = params2[1]

    # Test case 3: Gradient is None for one parameter
    params3 = [torch.ones(2, 2, requires_grad=True, device='cuda'), torch.zeros(3, requires_grad=True, device='cuda')]
    grads3 = [torch.full_like(params3[0], 2.0), None] # Grad for second param is None
    expected_params3_0 = params3[0].clone() - 0.1 * grads3[0]
    expected_params3_1 = params3[1].clone() # Should remain unchanged
    SGD_step(params3, grads3, lr=0.1)
    results["test_case_3_param0"] = params3[0]
    results["test_case_3_param1"] = params3[1] # Should be tensor of zeros


    return results

test_results = test_SGD()
print(test_results)