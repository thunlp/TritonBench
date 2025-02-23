import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.fused_layernorm_triton import fused_native_layer_norm
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_layernorm_triton', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):
            size = 2 ** i
            S = size // 1024  # Example division to get S and D
            D = 1024
            primals_1 = torch.rand(D, dtype=torch.bfloat16)
            primals_2 = torch.rand(D, dtype=torch.bfloat16)
            primals_3 = torch.rand(S, D, dtype=torch.bfloat16)
            self.input_tensors.append((primals_1, primals_2, primals_3))

    def to_cuda(self, input_tensor):
        return tuple(tensor.cuda() for tensor in input_tensor)

    def call_op(self, input_tensor):
        return fused_native_layer_norm(*input_tensor)

    def get_gbps(self, input_tensor, runtime):
        primals_3 = input_tensor[2]
        total_bytes = 3 * primals_3.numel() * primals_3.element_size()  # 3 tensors are involved
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        primals_3 = input_tensor[2]
        # Assuming each element involves a few operations, e.g., subtraction, multiplication, etc.
        FLOPS = 5 * primals_3.numel()  # Example: 5 operations per element
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
