import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.tril_mm_and_scale import tril_mm_and_scale  # 引入算子
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('tril_mm_and_scale', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 12):  # 测试不同规模：n从16到256
            n = 2 ** i
            p = n  # 保持B矩阵为方形，可根据需要调整
            A = torch.rand((n, n), dtype=self.dtype or torch.float32)
            B = torch.rand((n, p), dtype=self.dtype or torch.float32)
            alpha = 0.5
            beta = 0.5
            self.input_tensors.append((A, B, alpha, beta))  # 输入包含两个矩阵和标量参数

    def to_cuda(self, input_tensor):
        A, B, alpha, beta = input_tensor
        return (A.cuda(), B.cuda(), alpha, beta)  # 仅转移张量到CUDA

    def call_op(self, input_tensor):
        A, B, alpha, beta = input_tensor
        return tril_mm_and_scale(A, B, alpha, beta)  # 调用目标算子

    def get_gbps(self, input_tensor, runtime):
        A, B, _, _ = input_tensor
        n, p = A.shape[0], B.shape[1]
        element_size = A.element_size()
        # 总数据量 = 输入(A+B) + 输出(n x p矩阵)
        total_bytes = (A.numel() + B.numel() + n*p) * element_size
        return total_bytes / (runtime / 1000) / 1e9  # 转换为GB/s

    def get_tflops(self, input_tensor, runtime):
        A, B, _, _ = input_tensor
        n, p = A.shape[0], B.shape[1]
        # 计算理论FLOPs:
        # 1. 三角矩阵乘法：2*n^2*p (按全矩阵计算)
        # 2. alpha和beta缩放：2*n*p
        total_flops = 2 * n**2 * p + 2 * n * p
        return total_flops / (runtime / 1000) / 1e12  # 转换为TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
