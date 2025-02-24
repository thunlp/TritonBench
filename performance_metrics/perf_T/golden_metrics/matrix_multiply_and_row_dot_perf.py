import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.matrix_multiply_and_row_dot import matrix_multiply_and_row_dot
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('matrix_multiply_and_row_dot', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        # 生成不同规模的矩阵输入：A(n, m), B(m, p), C(n, p)
        for k in range(2, 33):
            size = 128 * k
            # 生成方阵便于测试
            A = torch.rand((size, size), dtype=torch.float16)
            B = torch.rand((size, size), dtype=torch.float16)
            C = torch.rand((size, size), dtype=torch.float16)
            alpha = 1.0
            beta = 1.0
            self.input_tensors.append((A, B, alpha, beta, C))

    def to_cuda(self, input_tuple):
        # 将输入元组中的张量移动到GPU
        A, B, alpha, beta, C = input_tuple
        return (A.cuda(), B.cuda(), alpha, beta, C.cuda())
    
    def call_op(self, input_tuple):
        # 调用算子函数
        A, B, alpha, beta, C = input_tuple
        return matrix_multiply_and_row_dot(A, B, alpha, beta, C)
    
    def get_gbps(self, input_tuple, runtime):
        # 计算总内存带宽 (GB/s)
        A, B, _, _, C = input_tuple
        n, m = A.shape
        _, p = B.shape
        
        # 计算总数据量:
        # 读取: A(n*m) + B(m*p) + C(n*p)
        # 写入: C(n*p)
        # 点积读取: C[0](p) + C[1](p)
        total_bytes = (n*m + m*p + 3*n*p + 2*p) * 2  # float16=2字节
        
        GBPS = (total_bytes / 1e9) / (runtime / 1000)
        return GBPS
    
    def get_tflops(self, input_tuple, runtime):
        # 计算理论计算量 (TFLOPS)
        A, B, _, _, _ = input_tuple
        n, m = A.shape
        _, p = B.shape
        
        # 矩阵乘法计算量: 2*n*m*p
        # 其他计算: alpha乘(n*p) + beta乘(n*p) + 加法(n*p) + 点积(2*p)
        flops = 2 * n * m * p + 3 * n * p + 2*p
        TFLOPS = flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
