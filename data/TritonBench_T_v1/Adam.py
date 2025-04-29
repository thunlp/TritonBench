import torch

def simple_adam_step(
        params: torch.Tensor,
        grads: torch.Tensor,
        lr: float = 0.001,
        eps: float = 1e-08,
        weight_decay: float = 0
        ) -> torch.Tensor:
    """
    Performs a single simplified step resembling the Adam optimizer update rule.
    This implementation omits the exponential moving averages (m, v) used in standard Adam,
    calculating the update based only on the current gradient.

    Args:
        params: Parameters to optimize.
        grads: Gradients of the parameters.
        lr: Learning rate.
        eps: Term added to the denominator to improve numerical stability.
        weight_decay: Weight decay (L2 penalty).

    Returns:
        Tensor: The updated parameters.
    """

    grad = grads

    if weight_decay != 0:
        grad = grad.add(params.detach(), alpha=weight_decay)

    m_hat = grad
    v_hat = grad * grad

    # Denominator term in the Adam update rule
    denom = torch.sqrt(v_hat).add_(eps)

    update_amount = lr * m_hat / denom

    new_params = params - update_amount

    return new_params

##################################################################################################################################################

import torch
torch.manual_seed(42)

def test_simple_adam_step():
    results = {}

    # Basic test case
    params = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda', requires_grad=True)
    grads = torch.tensor([[0.1, 0.2], [0.3, 0.4]], device='cuda')
    lr = 0.001
    eps = 1e-8

    updated_params = simple_adam_step(params.clone(), grads, lr=lr, eps=eps, weight_decay=0)

    # Check output shape and type
    results['basic_shape_match'] = updated_params.shape == params.shape
    results['basic_dtype_match'] = updated_params.dtype == params.dtype
    results['basic_device_match'] = updated_params.device == params.device

    # Check calculation (simplified for demonstration)
    expected_update = lr * torch.sign(grads)
    # Using a loose check for demonstration
    results['basic_calculation_approx_correct'] = torch.all(torch.abs((params - updated_params) - expected_update) < lr * 0.5).item()

    # Test with weight decay
    params_wd = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda', requires_grad=True)
    grads_wd = torch.tensor([[0.1, 0.2], [0.3, 0.4]], device='cuda')
    weight_decay = 0.01

    updated_params_wd = simple_adam_step(params_wd.clone(), grads_wd, lr=lr, eps=eps, weight_decay=weight_decay)

    # Check output shape and type for weight decay case
    results['wd_shape_match'] = updated_params_wd.shape == params_wd.shape
    results['wd_dtype_match'] = updated_params_wd.dtype == params_wd.dtype
    results['wd_device_match'] = updated_params_wd.device == params_wd.device

    # Check that weight decay modified the update
    results['wd_params_different_from_basic'] = not torch.allclose(updated_params_wd, updated_params)

    return results

# Run the tests and print the results dictionary
test_results = test_simple_adam_step()
print(test_results)