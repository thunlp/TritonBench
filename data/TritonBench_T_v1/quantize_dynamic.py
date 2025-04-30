import torch
from torch import nn
from typing import Optional, Any
from torch.quantization import quantize_dynamic

def dynamic_custom(
        model: torch.nn.Module, 
        qconfig_spec: Optional[dict[str, Any]]=None, 
        inplace: bool=False, 
        mapping: Optional[dict[str, Any]]=None) -> torch.nn.Module:
    """
    Custom wrapper to convert a float model to a dynamic quantized model by replacing specified modules
    with their dynamic weight-only quantized versions.
    
    Args:
        model (torch.nn.Module): Input model to be quantized.
        qconfig_spec (dict or set, optional): Either a dictionary mapping submodule names/types 
                                              to quantization configurations or a set of types/names 
                                              for dynamic quantization. Default is None.
        inplace (bool, optional): If True, the transformation is carried out in-place, mutating 
                                  the original module. Default is False.
        mapping (dict, optional): Maps submodule types to dynamically quantized versions. Default is None.
    
    Returns:
        torch.nn.Module: The dynamic quantized model.
    """    
    # Use the torch.quantization.quantize_dynamic function to avoid recursion
    quantized_model = quantize_dynamic(model, qconfig_spec=qconfig_spec, inplace=inplace, mapping=mapping)
    
    return quantized_model

##################################################################################################################################################

import torch
torch.manual_seed(42)

def test_quantize_dynamic():
    # Define a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(10, 5)

        def forward(self, x):
            return self.fc(x)

    # Initialize model and move to GPU
    model = SimpleModel().cuda()

    # Prepare input tensor
    input_tensor = torch.randn(1, 10).cuda()

    # Dictionary to store results
    results = {}

    # Test case 1: Default quantization
    quantized_model_1 = dynamic_custom(model)
    results["test_case_1"] = quantized_model_1(input_tensor)

    # Test case 2: Quantization with qconfig_spec
    qconfig_spec = {nn.Linear}
    quantized_model_2 = dynamic_custom(model, qconfig_spec=qconfig_spec)
    results["test_case_2"] = quantized_model_2(input_tensor)

    # Test case 3: In-place quantization
    model_copy = SimpleModel().cuda()
    quantized_model_3 = dynamic_custom(model_copy, inplace=True)
    results["test_case_3"] = quantized_model_3(input_tensor)

    # Test case 4: Quantization with mapping
    mapping = {nn.Linear: nn.quantized.dynamic.Linear}
    quantized_model_4 = dynamic_custom(model, mapping=mapping)
    results["test_case_4"] = quantized_model_4(input_tensor)

    return results

test_results = test_quantize_dynamic()
print(test_results)