import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.fused_mv_logsoftmax_dropout import fused_mv_logsoftmax_dropout
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_mv_logsoftmax_dropout', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 33):
            n = 128 * i
            m = 128 * i
            input_matrix = torch.rand(n, m, dtype=torch.float16)
            vec = torch.rand(m, dtype=torch.float16)
            self.input_tensors.append((input_matrix, vec))

    def to_cuda(self, input_tuple):
        input_matrix, vec = input_tuple
        return (input_matrix.cuda(), vec.cuda())
    
    def call_op(self, input_tuple):
        input_matrix, vec = input_tuple
        return fused_mv_logsoftmax_dropout(input_matrix, vec, p=0.5, training=True)
    
    def get_gbps(self, input_tuple, runtime):
        input_matrix, vec = input_tuple
        n, m = input_matrix.shape
        element_size = input_matrix.element_size()
        total_bytes = (n * m + m + n * 5) * element_size
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tuple, runtime):
        input_matrix, vec = input_tuple
        n, m = input_matrix.shape
        matmul_flops = 2 * n * m
        log_softmax_flops = 4 * n
        dropout_flops = n
        total_flops = matmul_flops + log_softmax_flops + dropout_flops
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
