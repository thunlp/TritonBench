import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.fused_mv_sigmoid_sub import fused_mv_sigmoid_sub
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_mv_sigmoid_sub', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for k in range(6, 16):
            n = 2 ** k
            input_matrix = torch.rand(n, n, dtype=torch.float32)
            vec = torch.rand(n, dtype=torch.float32)
            other = 0.5
            self.input_tensors.append((input_matrix, vec, other))

    def to_cuda(self, input_tuple):
        input_matrix, vec, other = input_tuple
        return (input_matrix.cuda(), vec.cuda(), other)

    def call_op(self, input_tuple):
        input_matrix, vec, other = input_tuple
        return fused_mv_sigmoid_sub(input_matrix, vec, other, alpha=1)

    def get_gbps(self, input_tuple, runtime):
        input_matrix, vec, other = input_tuple
        n, m = input_matrix.shape
        total_bytes = (n * m + m + n * 5) * input_matrix.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tuple, runtime):
        input_matrix, vec, other = input_tuple
        n, m = input_matrix.shape
        total_flops = 2 * n * m + 5 * n
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
