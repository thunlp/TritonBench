import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.matrix_multiply_symmetric import matrix_multiply_symmetric
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('matrix_multiply_symmetric', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 33):
            n = 128 * i
            m = n  # 保证A@B后能进行C@C.T运算
            A = torch.rand(n, m, dtype=torch.float16)
            B = torch.rand(m, n, dtype=torch.float16)
            C = torch.rand(n, n, dtype=torch.float16)
            alpha = 1.0
            beta = 0.5
            self.input_tensors.append((A, B, C, alpha, beta))

    def to_cuda(self, input_tuple):
        A, B, C, alpha, beta = input_tuple
        A = A.cuda()
        B = B.cuda()
        C = C.cuda()
        return (A, B, C, alpha, beta)

    def call_op(self, input_tuple):
        A, B, C, alpha, beta = input_tuple
        return matrix_multiply_symmetric(A, B, C, alpha, beta)

    def get_gbps(self, input_tuple, runtime):
        A, B, C, alpha, beta = input_tuple
        element_size = A.element_size()  # 所有张量dtype一致
        # 总数据量 = (A + B + C读 + C写)
        total_bytes = (A.numel() + B.numel() + 2 * C.numel()) * element_size
        GBPS = total_bytes / (runtime / 1000) / 1e9  # runtime单位ms转秒
        return GBPS

    def get_tflops(self, input_tuple, runtime):
        A, B, C, alpha, beta = input_tuple
        n, m = A.shape[0], A.shape[1]
        # 计算各阶段FLOPS
        flops_AB = 2 * n * m * n   # 第一次矩阵乘法
        flops_add1 = 3 * n * n     # 第一次线性组合
        flops_CCt = 2 * n ** 3     # 第二次矩阵乘法
        flops_add2 = 3 * n * n     # 第二次线性组合
        total_flops = flops_AB + flops_add1 + flops_CCt + flops_add2
        TFLOPS = total_flops / (runtime / 1000) / 1e12  # 转为TFLOPS
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
