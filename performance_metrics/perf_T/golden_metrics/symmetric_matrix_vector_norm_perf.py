import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.symmetric_matrix_vector_norm import symmetric_matrix_vector_norm
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, alpha=1.0, beta=1.0, p=2.0, **kwargs):
        super().__init__('symmetric_matrix_vector_norm', dtype=dtype, is_backward=is_backward, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.p = p

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 15):
            n = 2 ** i
            A = torch.rand((n, n), dtype=self.dtype)
            A = (A + A.T) / 2
            x = torch.rand(n, dtype=self.dtype)
            self.input_tensors.append((A, x))

    def to_cuda(self, input_tensor):
        A, x = input_tensor
        return (A.cuda(), x.cuda())
    
    def call_op(self, input_tensor):
        A, x = input_tensor
        return symmetric_matrix_vector_norm(A, x, self.alpha, self.beta, self.p)
    
    def get_gbps(self, input_tensor, runtime):
        A, x = input_tensor
        input_bytes = (A.numel() + x.numel()) * A.element_size()
        output_bytes = torch.tensor(0.0, dtype=A.dtype).element_size()
        total_bytes = input_bytes + output_bytes
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        A, x = input_tensor
        n = A.size(0)
        flops = 2 * n ** 2
        TFLOPS = flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
