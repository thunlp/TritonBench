import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.symmetric_mm_and_abs_sum import symmetric_mm_and_abs_sum
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('symmetric_mm_and_abs_sum', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 12):
            n = 2 ** i
            m = n  # 测试方阵情况
            A = torch.rand((n, m), dtype=self.dtype)
            C = torch.rand((n, n), dtype=self.dtype)
            alpha = 1.0
            beta = 0.5
            self.input_tensors.append((A, C, alpha, beta))

    def to_cuda(self, input_tuple):
        # 将张量转移到CUDA，保留标量参数
        A, C, alpha, beta = input_tuple
        return (A.cuda(), C.cuda(), alpha, beta)
        
    def call_op(self, input_tuple):
        # 解包参数并调用算子
        A, C, alpha, beta = input_tuple
        return symmetric_mm_and_abs_sum(A, C, alpha, beta)

    def get_gbps(self, input_tuple, runtime):
        # 计算内存带宽利用率
        A, C, _, _ = input_tuple
        bytes_A = A.numel() * A.element_size()
        bytes_C = C.numel() * C.element_size()
        total_bytes = bytes_A + 2 * bytes_C  # 读A+C，写C
        return total_bytes / (runtime / 1000) / 1e9  # 转换为GB/s

    def get_tflops(self, input_tuple, runtime):
        # 计算浮点运算吞吐量
        A, C, _, _ = input_tuple
        n, m = A.shape
        # 矩阵乘法核心运算量 (2*n*m*n)
        mm_flops = 2 * n * m * n
        # 其他运算量 (3n² + n² + n²-1)
        other_flops = 5 * n**2 - 1
        total_flops = mm_flops + other_flops
        return total_flops / (runtime / 1000) / 1e12  # 转换为TFLOP/s

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
