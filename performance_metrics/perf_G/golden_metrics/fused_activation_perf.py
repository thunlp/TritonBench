import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.fused_activation import fused_add_mul_activation_torch
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_activation', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 30):
            size = 2 ** i
            in_out_tensor = torch.rand(size, dtype=torch.float16)
            bias = torch.rand(size // 4, dtype=torch.float16)  # Assuming num_weights is size // 4
            in_tensor = torch.rand(size, dtype=torch.float16)
            self.input_tensors.append((in_out_tensor, bias, in_tensor))

    def to_cuda(self, input_tensor):
        in_out_tensor, bias, in_tensor = input_tensor
        return (in_out_tensor.cuda(), bias.cuda(), in_tensor.cuda())

    def call_op(self, input_tensor):
        in_out_tensor, bias, in_tensor = input_tensor
        return fused_add_mul_activation_torch(in_out_tensor, bias, in_tensor)

    def get_gbps(self, input_tensor, runtime):
        in_out_tensor, bias, in_tensor = input_tensor
        total_bytes = (in_out_tensor.numel() + bias.numel() + in_tensor.numel()) * in_out_tensor.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        in_out_tensor, bias, in_tensor = input_tensor
        FLOPS = 2 * in_out_tensor.numel()  # Assuming 2 operations per element (add and multiply)
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
