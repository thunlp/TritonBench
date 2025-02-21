import torch
import torch.nn as nn

def SGD(model, input, target, loss_fn, lr=0.1, momentum=0.9):
    """
    Performs a single step of SGD optimization.

    Args:
    - model (torch.nn.Module): The model to optimize.
    - input (torch.Tensor): The input tensor for the model.
    - target (torch.Tensor): The target tensor.
    - loss_fn (callable): The loss function.
    - lr (float, optional): The learning rate for the optimizer. Default is 0.1.
    - momentum (float, optional): The momentum for the optimizer. Default is 0.9.

    Returns:
    - loss (torch.Tensor): The computed loss for the step.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    optimizer.zero_grad()
    loss = loss_fn(model(input), target)
    loss.backward()
    optimizer.step()
    return loss

##################################################################################################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

def test_SGD():
    results = {}
    
    # Test case 1: Basic functionality
    model = SimpleModel().cuda()
    input = torch.randn(5, 10).cuda()
    target = torch.randn(5, 1).cuda()
    loss_fn = nn.MSELoss()
    loss = SGD(model, input, target, loss_fn)
    results["test_case_1"] = loss.item()

    # Test case 2: Different learning rate
    model = SimpleModel().cuda()
    input = torch.randn(5, 10).cuda()
    target = torch.randn(5, 1).cuda()
    loss_fn = nn.MSELoss()
    loss = SGD(model, input, target, loss_fn, lr=0.01)
    results["test_case_2"] = loss.item()

    # Test case 3: Different momentum
    model = SimpleModel().cuda()
    input = torch.randn(5, 10).cuda()
    target = torch.randn(5, 1).cuda()
    loss_fn = nn.MSELoss()
    loss = SGD(model, input, target, loss_fn, momentum=0.5)
    results["test_case_3"] = loss.item()

    # Test case 4: Different loss function
    model = SimpleModel().cuda()
    input = torch.randn(5, 10).cuda()
    target = torch.randint(0, 2, (5, 1)).float().cuda()
    loss_fn = nn.BCEWithLogitsLoss()
    loss = SGD(model, input, target, loss_fn)
    results["test_case_4"] = loss.item()

    return results

test_results = test_SGD()
