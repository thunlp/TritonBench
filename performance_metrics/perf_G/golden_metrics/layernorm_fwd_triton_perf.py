import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.layernorm_fwd_triton import layernorm_forward
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('layernorm_fwd_triton', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 18):  # Adjust the range as needed for testing
            size = 2 ** i
            # Create 3D tensors for X and 2D tensor for W
            X = torch.rand((2048, 8, size), dtype=torch.float32)  # Example dimensions
            W = torch.rand((8, size), dtype=torch.float32)
            self.input_tensors.append((X, W))

    def to_cuda(self, input_tensor):
        X, W = input_tensor
        return (X.cuda(), W.cuda())

    def call_op(self, input_tensor):
        X, W = input_tensor
        eps = 1e-5  # Example epsilon value
        return layernorm_forward(X, W, eps)

    def get_gbps(self, input_tensor, runtime):
        X, W = input_tensor
        total_bytes = (X.numel() + W.numel() + X.numel()) * X.element_size()  # Input, weights, and output
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        X, _ = input_tensor
        # Assuming each element involves a few FLOPs: mean, variance, normalization, and scaling
        FLOPS = 5 * X.numel()  # Example estimation
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
