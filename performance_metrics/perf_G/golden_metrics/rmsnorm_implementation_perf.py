import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.rmsnorm_implementation import rmsnorm_wrapper
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('rmsnorm_implementation', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 11):  # Adjust the range as needed for your testing
            batch_size = 2 ** i
            M = 128  # Fixed M dimension, can be adjusted
            K = 4096  # Fixed K dimension, can be adjusted
            x = torch.rand((batch_size, M, K), dtype=torch.float16)
            rms_weights = torch.rand((K,), dtype=torch.float16)
            self.input_tensors.append((x, rms_weights))

    def to_cuda(self, input_tensor):
        x, rms_weights = input_tensor
        return (x.cuda(), rms_weights.cuda())

    def call_op(self, input_tensor):
        x, rms_weights = input_tensor
        return rmsnorm_wrapper(x, rms_weights)

    def get_gbps(self, input_tensor, runtime):
        x, _ = input_tensor
        total_bytes = (2 * x.numel() + x.size(-1)) * x.element_size()  # Read x and rms_weights, write out
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        x, _ = input_tensor
        FLOPS = 2 * x.numel()  # Each element involves a division and a multiplication
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
