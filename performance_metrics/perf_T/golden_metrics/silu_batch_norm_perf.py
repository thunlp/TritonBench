import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.silu_batch_norm import silu_batch_norm
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('silu_batch_norm', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 16):
            print(i)
            size = 2 ** i
            input_tensor = torch.randn(16, size, 8, 128, dtype=self.dtype)
            running_mean = torch.zeros(size, dtype=self.dtype)
            running_var = torch.ones(size, dtype=self.dtype)
            weight = torch.randn(size, dtype=self.dtype)
            bias = torch.randn(size, dtype=self.dtype)
            self.input_tensors.append((input_tensor, running_mean, running_var, weight, bias))
    
    def to_cuda(self, input_tuple):
        input_tensor, running_mean, running_var, weight, bias = input_tuple
        input_tensor = input_tensor.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
        weight = weight.cuda()
        bias = bias.cuda()
        return (input_tensor, running_mean, running_var, weight, bias)
        
    def call_op(self, input_tuple):
        input_tensor, running_mean, running_var, weight, bias = input_tuple
        return silu_batch_norm(
            input_tensor, 
            running_mean, 
            running_var, 
            weight, 
            bias, 
            training=False,
            momentum=0.1, 
            eps=1e-5
        )
    
    def get_gbps(self, input_tuple, runtime):
        input_tensor, running_mean, running_var, weight, bias = input_tuple
        bytes_input = input_tensor.numel() * input_tensor.element_size()
        bytes_running_mean = running_mean.numel() * running_mean.element_size()
        bytes_running_var = running_var.numel() * running_var.element_size()
        bytes_weight = weight.numel() * weight.element_size()
        bytes_bias = bias.numel() * bias.element_size()
        bytes_output = input_tensor.numel() * input_tensor.element_size()
        
        total_bytes = (bytes_input + bytes_running_mean + bytes_running_var +
                       bytes_weight + bytes_bias + bytes_output)
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tuple, runtime):
        input_tensor = input_tuple[0]
        total_flops = 10 * input_tensor.numel()
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
