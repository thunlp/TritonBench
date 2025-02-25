import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.sigmoid_batch_norm import sigmoid_batch_norm
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('sigmoid_batch_norm', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):
            size = 2 ** i
            batch_size = 32
            channels = size // batch_size
            
            input_tensor = torch.randn(batch_size, channels, dtype=torch.float32)
            
            running_mean = torch.randn(channels)
            running_var = torch.abs(torch.randn(channels)) + 1e-5
            weight = torch.randn(channels)
            bias = torch.randn(channels)
            
            self.input_tensors.append((
                input_tensor,
                running_mean,
                running_var,
                weight,
                bias
            ))

    def to_cuda(self, input_tuple):
        return tuple(t.cuda() if isinstance(t, torch.Tensor) else t for t in input_tuple)
    
    def call_op(self, input_tuple):
        input_tensor, running_mean, running_var, weight, bias = input_tuple
        return sigmoid_batch_norm(
            input=input_tensor,
            running_mean=running_mean,
            running_var=running_var,
            weight=weight,
            bias=bias,
            training=False
        )
    
    def get_gbps(self, input_tuple, runtime):
        input_tensor, running_mean, running_var, weight, bias = input_tuple
        element_size = input_tensor.element_size()
        
        input_bytes = input_tensor.numel() * element_size
        running_mean_bytes = running_mean.numel() * element_size
        running_var_bytes = running_var.numel() * element_size
        weight_bytes = weight.numel() * element_size
        bias_bytes = bias.numel() * element_size
        
        output_bytes = input_tensor.numel() * element_size
        
        total_bytes = input_bytes + running_mean_bytes + running_var_bytes + weight_bytes + bias_bytes + output_bytes * 3
        return total_bytes / (runtime / 1000) / 1e9  # GB/s

    def get_tflops(self, input_tuple, runtime):
        input_tensor, _, _, _, _ = input_tuple
        flops_per_element = 10
        total_flops = input_tensor.numel() * flops_per_element
        return total_flops / (runtime / 1000) / 1e12  # TFLOP/s

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
