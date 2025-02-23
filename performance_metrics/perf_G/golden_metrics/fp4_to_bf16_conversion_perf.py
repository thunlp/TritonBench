import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.fp4_to_bf16_conversion import triton_f4_to_scaled_bf16
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fp4_to_bf16_conversion', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):
            size = 2 ** i
            x = torch.randint(0, 16, (size,), dtype=torch.uint8)  # Packed fp4 values
            s_e8m0 = torch.randint(0, 256, (size // 32,), dtype=torch.uint8)  # Scale in e8m0 format
            self.input_tensors.append((x, s_e8m0))

    def to_cuda(self, input_tensor):
        x, s_e8m0 = input_tensor
        return x.cuda(), s_e8m0.cuda()

    def call_op(self, input_tensor):
        x, s_e8m0 = input_tensor
        mx_block_size = 32  # Assuming block size of 32 as per the function description
        return triton_f4_to_scaled_bf16(x, s_e8m0, mx_block_size)

    def get_gbps(self, input_tensor, runtime):
        x, _ = input_tensor
        total_bytes = x.numel() * x.element_size() * 2  # Input and output
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        x, _ = input_tensor
        # Assuming each conversion involves a few operations, adjust as necessary
        operations_per_element = 5  # Hypothetical number of operations per element
        FLOPS = operations_per_element * x.numel()
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
