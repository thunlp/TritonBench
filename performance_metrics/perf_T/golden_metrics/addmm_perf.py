import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.addmm import addmm
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('addmm', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 33):
            size = 128 * i
            input_tensor = torch.rand((size, size), dtype=torch.float16)
            mat1 = torch.rand((size, size), dtype=torch.float16)
            mat2 = torch.rand((size, size), dtype=torch.float16)
            self.input_tensors.append((input_tensor, mat1, mat2))

    def to_cuda(self, input_tensor_tuple):
        return tuple(t.cuda() for t in input_tensor_tuple)
    
    def call_op(self, input_tensor_tuple):
        input, mat1, mat2 = input_tensor_tuple
        return addmm(input, mat1, mat2)
    
    def get_gbps(self, input_tensor_tuple, runtime):
        input, mat1, mat2 = input_tensor_tuple
        M, N = input.shape
        K = mat1.shape[1]
        element_size = input.element_size()
        
        total_bytes = (M*N + M*K + K*N + M*N) * element_size
        return total_bytes / (runtime / 1000) / 1e9  # GBPS

    def get_tflops(self, input_tensor_tuple, runtime):
        input, mat1, mat2 = input_tensor_tuple
        M, N = input.shape
        K = mat1.shape[1]
        flops = 2 * M * N * K
        return flops / (runtime / 1000) / 1e12  # TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
