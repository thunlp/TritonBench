import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Correctly import the layer norm function
from TritonBench_v1.layer_norm_liger import LigerLayerNormFunction
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('layer_norm_liger', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 20):  # Example sizes from 2^12 to 2^27
            size = 2 ** i
            n_cols = 128  # Example feature dimension
            X = torch.rand(size, n_cols, dtype=torch.float32)
            W = torch.rand(n_cols, dtype=torch.float32)
            B = torch.rand(n_cols, dtype=torch.float32)
            self.input_tensors.append((X, W, B))

    def to_cuda(self, input_tensor):
        X, W, B = input_tensor
        return (X.cuda(), W.cuda(), B.cuda())

    def call_op(self, input_tensor):
        X, W, B = input_tensor
        eps = 1e-5  # Example epsilon value for numerical stability
        return LigerLayerNormFunction.apply(X, W, B, eps)

    def get_gbps(self, input_tensor, runtime):
        X, W, B = input_tensor
        total_bytes = (X.numel() + W.numel() + B.numel()) * X.element_size() * 2  # Read and write
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        X, W, B = input_tensor
        # Assuming each element involves a few FLOPs (e.g., multiply, add)
        FLOPS = 5 * X.numel()  # Example: 5 operations per element
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
