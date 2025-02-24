import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.bgmv_shrink_kernel import _bgmv_shrink
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('bgmv_shrink_kernel', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 9):  # Adjust the range for different sizes
            size = 2 ** i
            inputs = torch.rand(size, 256, dtype=torch.float16)  # Example dimensions
            lora_a_weights = torch.rand(1, size, 256, dtype=torch.float16)  # Example dimensions
            output_tensor = torch.zeros(size, 256, dtype=torch.float16)  # Example dimensions
            lora_indices_tensor = torch.randint(0, 1, (size,), dtype=torch.int32)  # Example indices
            self.input_tensors.append((inputs, lora_a_weights, output_tensor, lora_indices_tensor))

    def to_cuda(self, input_tensor):
        inputs, lora_a_weights, output_tensor, lora_indices_tensor = input_tensor
        return (inputs.cuda(), lora_a_weights.cuda(), output_tensor.cuda(), lora_indices_tensor.cuda())

    def call_op(self, input_tensor):
        inputs, lora_a_weights, output_tensor, lora_indices_tensor = input_tensor
        _bgmv_shrink(inputs, lora_a_weights, output_tensor, lora_indices_tensor, scaling=1.0)
        return output_tensor

    def get_gbps(self, input_tensor, runtime):
        inputs, lora_a_weights, output_tensor, lora_indices_tensor = input_tensor
        total_bytes = (inputs.numel() + lora_a_weights.numel() + output_tensor.numel()) * inputs.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        inputs, lora_a_weights, output_tensor, lora_indices_tensor = input_tensor
        FLOPS = 2 * inputs.size(0) * inputs.size(1) * lora_a_weights.size(-1)  # Example FLOPS calculation
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
