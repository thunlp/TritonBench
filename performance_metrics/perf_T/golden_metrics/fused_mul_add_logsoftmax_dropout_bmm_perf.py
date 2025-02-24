import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.fused_mul_add_logsoftmax_dropout_bmm import fused_mul_add_logsoftmax_dropout_bmm
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_mul_add_logsoftmax_dropout_bmm', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        B = 32  # Fixed batch size
        for i in range(5, 20):  # Generates sizes from 32x32 to 2048x2048
            M = 128 * i
            N = 128 * i
            P = 128 * i
            input1 = torch.randn(B, M, N, dtype=self.dtype)
            input2 = torch.randn(B, M, N, dtype=self.dtype)
            other = torch.randn(B, M, N, dtype=self.dtype)
            mat2 = torch.randn(B, N, P, dtype=self.dtype)
            self.input_tensors.append((input1, input2, other, mat2))

    def to_cuda(self, input_tuple):
        input1, input2, other, mat2 = input_tuple
        return (input1.cuda(), input2.cuda(), other.cuda(), mat2.cuda())
    
    def call_op(self, input_tuple):
        input1, input2, other, mat2 = input_tuple
        return fused_mul_add_logsoftmax_dropout_bmm(
            input1, input2, other, mat2, 
            p=0.5, training=True, inplace=False, dim=-1
        )
    
    def get_gbps(self, input_tuple, runtime):
        input1, input2, other, mat2 = input_tuple
        B, M, N = input1.shape[0], input1.shape[1], input1.shape[2]
        P = mat2.shape[2]
        element_size = input1.element_size()
        
        # Calculate total bytes transferred (inputs + outputs)
        input_bytes = (input1.numel() + input2.numel() + other.numel() + mat2.numel()) * element_size
        output_bytes = B * M * P * element_size
        total_bytes = input_bytes + output_bytes * 10
        
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tuple, runtime):
        input1, input2, other, mat2 = input_tuple
        B, M, N = input1.shape[0], input1.shape[1], input1.shape[2]
        P = mat2.shape[2]
        
        # Calculate FLOPs for each operation
        flops_mul = B * M * N
        flops_add = B * M * N
        flops_logsoftmax = 3 * B * M * N  # Estimated 3 FLOPs per element
        flops_dropout = B * M * N
        flops_bmm = 2 * B * M * N * P
        
        total_flops = flops_mul + flops_add + flops_logsoftmax + flops_dropout + flops_bmm
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
