import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.fused_transformer_block import fused_transformer_block
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_transformer_block', dtype=dtype, is_backward=is_backward, **kwargs)
        self.dtype = dtype if dtype is not None else torch.float32

    def get_input_tensors(self):
        self.input_tensors = []
        batch_size = 8
        seq_len = 128
        
        for exp in range(8, 13):
            d_in = 2 ** exp
            d_k = 4 * d_in
            d_out = d_in
            
            input_tensor = torch.randn(batch_size, seq_len, d_in, dtype=self.dtype)
            weight1 = torch.randn(d_in, d_k, dtype=self.dtype)
            weight2 = torch.randn(d_k, d_out, dtype=self.dtype)
            residual = torch.randn(batch_size, seq_len, d_out, dtype=self.dtype)
            
            self.input_tensors.append((input_tensor, weight1, weight2, residual))

    def to_cuda(self, input_tuple):
        return tuple(t.cuda() for t in input_tuple)
    
    def call_op(self, input_tuple):
        input_tensor, weight1, weight2, residual = input_tuple
        return fused_transformer_block(input_tensor, weight1, weight2, residual)
    
    def get_gbps(self, input_tuple, runtime):
        input_tensor, weight1, weight2, residual = input_tuple
        B, N, D_in = input_tensor.shape
        D_out = weight2.shape[1]
        
        element_size = input_tensor.element_size()
        input_bytes = input_tensor.numel() * element_size
        w1_bytes = weight1.numel() * element_size
        w2_bytes = weight2.numel() * element_size
        residual_bytes = residual.numel() * element_size
        output_bytes = B * N * D_out * element_size
        
        total_bytes = input_bytes + w1_bytes + w2_bytes + residual_bytes + output_bytes
        
        return total_bytes / (runtime / 1000) / 1e9

    def get_tflops(self, input_tuple, runtime):
        input_tensor, weight1, weight2, residual = input_tuple
        B, N, D_in = input_tensor.shape
        D_k = weight1.shape[1]
        D_out = weight2.shape[1]
        
        flops_z1 = 2 * B * N * D_in * D_k
        flops_softmax = 3 * B * N * D_k
        flops_dropout = 1 * B * N * D_k
        flops_z4 = 2 * B * N * D_k * D_out
        flops_residual = 1 * B * N * D_out
        flops_ln = 8 * B * N * D_out
        
        total_flops = sum([
            flops_z1, flops_softmax, flops_dropout,
            flops_z4, flops_residual, flops_ln
        ])
        
        return total_flops / (runtime / 1000) / 1e12

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
