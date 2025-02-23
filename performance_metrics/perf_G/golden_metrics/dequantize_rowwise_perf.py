import sys
import os
import math
import torch
import triton
import triton.language as tl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Correctly import the operator
from TritonBench_v1.dequantize_rowwise import dequantize_rowwise
from performance_utils import Performance_Metrics, do_bench_config

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('dequantize_rowwise', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 20):
            rows = 2 ** i
            cols = 128  # Assuming a fixed block size for simplicity
            x = torch.randint(0, 256, (rows, cols), dtype=torch.uint8)
            state_x = torch.rand(rows, dtype=torch.float16)
            self.input_tensors.append((x, state_x))

    def to_cuda(self, input_tensor):
        x, state_x = input_tensor
        return (x.cuda(), state_x.cuda())

    def call_op(self, input_tensor):
        x, state_x = input_tensor
        return dequantize_rowwise(x, state_x)

    def get_gbps(self, input_tensor, runtime):
        x, _ = input_tensor
        total_bytes = x.numel() * x.element_size() + x.shape[0] * 4
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        x, _ = input_tensor
        FLOPS = 2 * x.numel()  # Each element involves a multiply and an add
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
