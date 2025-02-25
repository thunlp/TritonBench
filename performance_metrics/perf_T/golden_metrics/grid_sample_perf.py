import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.grid_sample import grid_sample
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('grid_sample', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(8, 12):
            H = W = 2 ** i
            C = 3
            N = 16
            
            input_tensor = torch.randn(N, C, H, W, dtype=torch.float32)
            
            H_out = H // 2
            W_out = W // 2
            grid_tensor = torch.rand(N, H_out, W_out, 2, dtype=torch.float32)
            grid_tensor = grid_tensor * 2 - 1
            
            self.input_tensors.append((input_tensor, grid_tensor))

    def to_cuda(self, input_tensor_tuple):
        input_tensor, grid_tensor = input_tensor_tuple
        return (input_tensor.cuda(), grid_tensor.cuda())
    
    def call_op(self, input_tensor_tuple):
        input_tensor, grid_tensor = input_tensor_tuple
        return grid_sample(input_tensor, grid_tensor)

    def get_gbps(self, input_tensor_tuple, runtime):
        input_tensor, grid_tensor = input_tensor_tuple
        element_size = input_tensor.element_size()
        
        input_bytes = input_tensor.numel() * element_size
        grid_bytes = grid_tensor.numel() * element_size
        
        N, C = input_tensor.shape[0], input_tensor.shape[1]
        H_out, W_out = grid_tensor.shape[1], grid_tensor.shape[2]
        output_bytes = N * C * H_out * W_out * element_size
        
        total_bytes = input_bytes + grid_bytes + output_bytes
        return total_bytes / (runtime / 1000) / 1e9

    def get_tflops(self, input_tensor_tuple, runtime):
        input_tensor, grid_tensor = input_tensor_tuple
        N, C = input_tensor.shape[0], input_tensor.shape[1]
        H_out, W_out = grid_tensor.shape[1], grid_tensor.shape[2]
        
        flops_per_sample = 7
        total_flops = N * C * H_out * W_out * flops_per_sample
        
        return total_flops / (runtime / 1000) / 1e12


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
