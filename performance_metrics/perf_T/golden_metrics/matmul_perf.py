import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.matmul import matmul
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('matmul', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 33):
            size = 128 * i
            tensor1 = torch.rand(size, size, dtype=torch.float16)
            tensor2 = torch.rand(size, size, dtype=torch.float16)
            self.input_tensors.append((tensor1, tensor2))

    def to_cuda(self, input_tensor):
        tensor1, tensor2 = input_tensor
        return (tensor1.cuda(), tensor2.cuda())
    
    def call_op(self, input_tensor):
        tensor1, tensor2 = input_tensor
        return matmul(tensor1, tensor2)
    
    def get_gbps(self, input_tensor, runtime):
        tensor1, tensor2 = input_tensor
        M, K = tensor1.shape
        _, N = tensor2.shape
        element_size = tensor1.element_size()
        
        total_bytes = (M*K + K*N + M*N) * element_size
        
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        tensor1, tensor2 = input_tensor
        M, K = tensor1.shape
        _, N = tensor2.shape
        
        flops = 2 * M * N * K
        TFLOPS = flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
