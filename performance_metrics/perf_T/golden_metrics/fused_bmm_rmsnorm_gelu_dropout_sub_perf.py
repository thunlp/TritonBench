import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.fused_bmm_rmsnorm_gelu_dropout_sub import fused_bmm_rmsnorm_gelu_dropout_sub
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_bmm_rmsnorm_gelu_dropout_sub', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        # 生成不同规模的输入张量
        for i in range(2, 20):  # 调整范围避免内存溢出
            B = 32               # 固定batch_size
            M = 128 * i
            K = 128 * i
            N = 128 * i
            # 生成输入张量 (保持CPU类型)
            input1 = torch.randn(B, M, K, dtype=self.dtype)
            input2 = torch.randn(B, K, N, dtype=self.dtype)
            other = torch.randn(B, M, N, dtype=self.dtype)
            normalized_shape = N  # 归一化最后一个维度
            self.input_tensors.append((input1, input2, other, normalized_shape))

    def to_cuda(self, input_tuple):
        # 将每个张量移动到CUDA
        input1, input2, other, normalized_shape = input_tuple
        return (input1.cuda(), input2.cuda(), other.cuda(), normalized_shape)
    
    def call_op(self, input_tuple):
        # 调用融合算子
        input1, input2, other, normalized_shape = input_tuple
        return fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape)
    
    def get_gbps(self, input_tuple, runtime):
        # 计算内存带宽 (GB/s)
        input1, input2, other, _ = input_tuple
        B, M, K = input1.shape
        _, _, N = input2.shape
        
        # 总数据量 = 输入 + 输出
        total_bytes = (input1.numel() + input2.numel() + other.numel() + B*M*N) * input1.element_size() + B*M*N * input1.element_size() * 6
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tuple, runtime):
        # 计算计算吞吐量 (TFLOPs)
        input1, input2, _, _ = input_tuple
        B, M, K = input1.shape
        _, _, N = input2.shape
        
        # 分解各操作FLOPs
        flops_bmm = B * M * N * K * 2    # BMM: 2*K flops per element
        flops_norm = B * M * N * 3       # RMSNorm: 3 flops per element
        flops_gelu = B * M * N * 5       # GELU: 5 flops per element
        
        total_flops = flops_bmm + flops_norm + flops_gelu
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
