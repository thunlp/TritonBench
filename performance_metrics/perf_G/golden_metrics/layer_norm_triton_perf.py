import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Correctly import the operator
from TritonBench_v1.layer_norm_triton import layer_norm
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('layer_norm_triton', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 15):
            size = 2 ** i
            normalized_shape = (size,)
            weight = torch.rand(size, dtype=torch.float32)
            bias = torch.rand(size, dtype=torch.float32)
            input_tensor = (torch.rand(size, dtype=torch.float32), normalized_shape, weight, bias, 1e-5)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        x, normalized_shape, weight, bias, eps = input_tensor
        return (x.cuda(), normalized_shape, weight.cuda(), bias.cuda(), eps)

    def call_op(self, input_tensor):
        x, normalized_shape, weight, bias, eps = input_tensor
        return layer_norm(x, normalized_shape, weight, bias, eps)

    def get_gbps(self, input_tensor, runtime):
        x = input_tensor[0]
        total_bytes = 2 * x.numel() * x.element_size()  # Read and write
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        x = input_tensor[0]
        # Assuming each element involves a few FLOPs for normalization
        FLOPS = 5 * float(x.numel())  # Approximation
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
