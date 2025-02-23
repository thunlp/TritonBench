import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.cholesky_solve import cholesky_solve
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('cholesky_solve', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        # 生成不同大小的输入：n从16到512，k固定为10
        for i in range(4, 16):  # n=16, 32, 64, 128, 256, 512
            n = 2 ** i
            k = 10
            # 生成正定的下三角矩阵L
            L = torch.rand(n, n, dtype=self.dtype)
            L = torch.tril(L)  # 下三角
            L.diagonal().add_(1e-3)  # 确保对角线元素足够大
            # 生成右侧矩阵B
            B = torch.rand(n, k, dtype=self.dtype)
            self.input_tensors.append((B, L))
    
    def to_cuda(self, input_tensor):
        B, L = input_tensor
        return (B.cuda(), L.cuda())
        
    def call_op(self, input_tensor):
        B, L = input_tensor
        return cholesky_solve(B, L, upper=False)  # L为下三角
    
    def get_gbps(self, input_tensor, runtime):
        B, L = input_tensor
        # 计算总数据量：B和L的读取 + X的写入
        B_size = B.numel() * B.element_size()
        L_size = L.numel() * L.element_size()
        X_size = B.numel() * B.element_size()  # X与B同shape
        total_bytes = B_size + L_size + X_size
        GBPS = total_bytes / (runtime / 1000) / 1e9  # 转为GB/s
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        B, L = input_tensor
        n, k = B.shape[-2], B.shape[-1]  # 提取矩阵维度
        # 计算浮点运算量：2*n²*k (前向+反向替换)
        FLOPS = 2 * (n ** 2) * k
        TFLOPS = FLOPS / (runtime / 1000) / 1e12  # 转为TFLOPS
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
