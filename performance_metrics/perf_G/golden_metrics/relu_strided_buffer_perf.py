import sys
import os
import math
from typing import Union
import torch
import triton
import triton.language as tl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Correctly import the operator
from TritonBench_v1.relu_strided_buffer import relu_forward_wrapper_rank_1
from performance_utils import Performance_Metrics, do_bench_config

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('relu_strided_buffer', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):
            size = 2 ** i
            in0 = torch.rand(size, dtype=torch.float16)
            out0 = torch.empty_like(in0)
            input_tensor = (in0, out0)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        in0, out0 = input_tensor
        in0 = in0.cuda()
        out0 = out0.cuda()
        return (in0, out0)

    def call_op(self, input_tensor):
        in0, out0 = input_tensor
        relu_forward_wrapper_rank_1(in0, out0=out0)

    def get_gbps(self, input_tensor, runtime):
        in0, out0 = input_tensor
        total_bytes = in0.numel() * in0.element_size() + out0.numel() * out0.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        in0, out0 = input_tensor
        FLOPS = in0.numel() * 1
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
