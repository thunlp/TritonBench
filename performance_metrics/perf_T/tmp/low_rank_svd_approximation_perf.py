import sys
import os
import json

sys.path.append('/home/lishangzhan/triton/torch_performance/GPU_efficiency/output_DeepSeek-R1')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from low_rank_svd_approximation import low_rank_svd_approximation
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('low_rank_svd_approximation', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        # 生成不同大小的方阵和对应的k值
        for i in range(2, 12):  # 256x256 到 8192x8192
            size = 128 * i
            A = torch.rand(size, size, dtype=self.dtype or torch.float32)
            k = min(50, size)  # 固定k=50（确保不超过矩阵尺寸）
            self.input_tensors.append((A, k))

    def to_cuda(self, input_tuple):
        A, k = input_tuple
        return (A.cuda(), k)  # 仅转移张量到CUDA，k保持不变

    def call_op(self, input_tuple):
        A, k = input_tuple
        return low_rank_svd_approximation(A, k)  # 调用算子

    def get_gbps(self, input_tuple, runtime):
        A, k = input_tuple
        # 输入和输出各占A.numel()，总数据量翻倍
        total_bytes = A.numel() * A.element_size() * 2
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tuple, runtime):
        A, k = input_tuple
        m, n = A.shape[-2], A.shape[-1]
        # 估算SVD的FLOPs（假设为2*m*n^2）
        flops_svd = 2 * m * n**2
        # 两次矩阵乘法的FLOPs
        flops_matmul = 2 * m * k * (k + n)
        total_flops = flops_svd + flops_matmul
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
