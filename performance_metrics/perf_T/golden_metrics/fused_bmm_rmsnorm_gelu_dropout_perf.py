import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.fused_bmm_rmsnorm_gelu_dropout import fused_bmm_rmsnorm_gelu_dropout
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_bmm_rmsnorm_gelu_dropout', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        # 不同规模的测试用例 (batch_size, M, N, K)
        for exp in range(2, 20):
            B = 32
            N, M, P = 128 * exp, 128 * exp, 128 * exp
            input1 = torch.randn(B, N, M, dtype=self.dtype)
            input2 = torch.randn(B, M, P, dtype=self.dtype)
            normalized_shape = P
            self.input_tensors.append((input1, input2, normalized_shape))

    def to_cuda(self, input_tuple):
        input1, input2, normalized_shape = input_tuple
        return (input1.cuda(), input2.cuda(), normalized_shape)
    
    def call_op(self, input_tuple):
        input1, input2, normalized_shape = input_tuple
        return fused_bmm_rmsnorm_gelu_dropout(
            input1, input2, 
            normalized_shape=normalized_shape,
            dropout_p=0.1,
            training=True  # 确保Dropout生效
        )
    
    def get_gbps(self, input_tuple, runtime):
        input1, input2, _ = input_tuple
        # 计算总数据传输量: input1 + input2 + output
        output_shape = (input1.shape[0], input1.shape[1], input2.shape[2])
        total_bytes = (input1.numel() + input2.numel() + output_shape[0]*output_shape[1]*output_shape[2]) * input1.element_size() + output_shape[0]*output_shape[1]*output_shape[2] * input1.element_size() * 6
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tuple, runtime):
        input1, input2, _ = input_tuple
        B, M, N = input1.shape
        _, _, K = input2.shape
        
        # 主要计算来自BMM操作
        bmm_flops = 2 * B * M * N * K
        # RMSNorm近似计算量 (2*B*M*K)
        # GELU近似计算量 (4*B*M*K)
        # Dropout近似计算量 (B*M*K)
        total_flops = bmm_flops + 7 * B * M * K
        
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
