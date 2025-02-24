import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.layer_norm_welfold import fused_native_layer_norm_no_welford
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('layer_norm_welfold', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        for i in range(4, 20):  # Adjust the range for desired sizes
            S = 2 ** i
            D = 1024  # Fixed dimension size for testing
            primals_1 = torch.rand(D, dtype=torch.float32)
            primals_2 = torch.rand(D, dtype=torch.float32)
            primals_3 = torch.rand(S, D, dtype=torch.bfloat16)
            self.input_tensors.append((primals_1, primals_2, primals_3))

    def to_cuda(self, input_tensor):
        return tuple(tensor.cuda() for tensor in input_tensor)

    def call_op(self, input_tensor):
        return fused_native_layer_norm_no_welford(*input_tensor)

    def get_gbps(self, input_tensor, runtime):
        primals_3 = input_tensor[2]
        total_bytes = (primals_3.numel() * primals_3.element_size() * 2)
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        primals_3 = input_tensor[2]
        FLOPS = 2 * primals_3.numel()  # Assuming 2 FLOPS per element
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
