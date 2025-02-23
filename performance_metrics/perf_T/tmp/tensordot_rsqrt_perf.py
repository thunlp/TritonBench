import sys
import os
import json

sys.path.append('/home/lishangzhan/triton/torch_performance/GPU_efficiency/output_DeepSeek-R1')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensordot_rsqrt import tensordot_rsqrt
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('tensordot_rsqrt', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        # 生成不同大小的二维方阵对，dims=1 表示矩阵乘法
        for i in range(5, 12):  # 控制矩阵尺寸在 256x256 到 8192x8192 之间
            size = 2 ** i
            a = torch.rand(size, size, dtype=self.dtype or torch.float32)
            b = torch.rand(size, size, dtype=self.dtype or torch.float32)
            dims = 1  # 固定收缩维度为最后一个维度
            self.input_tensors.append((a, b, dims))
    
    def to_cuda(self, input_tuple):
        a, b, dims = input_tuple
        return (a.cuda(), b.cuda(), dims)
    
    def call_op(self, input_tuple):
        a, b, dims = input_tuple
        return tensordot_rsqrt(a, b, dims)
    
    def get_gbps(self, input_tuple, runtime):
        a, b, dims = input_tuple
        # 计算总数据传输量（输入+输出）
        input_bytes = a.numel() * a.element_size() + b.numel() * b.element_size()
        output_elements = a.shape[0] * b.shape[1]  # 矩阵乘法结果形状
        output_bytes = output_elements * a.element_size()
        total_bytes = input_bytes + output_bytes
        return total_bytes / (runtime / 1000) / 1e9  # 转换为 GB/s
    
    def get_tflops(self, input_tuple, runtime):
        a, b, dims = input_tuple
        # 计算浮点运算量
        M, N = a.shape
        _, K = b.shape
        matrix_mult_flops = 2 * M * N * K    # 矩阵乘法 FLOPs
        rsqrt_flops = M * K                  # rsqrt 逐元素运算
        total_flops = matrix_mult_flops + rsqrt_flops
        return total_flops / (runtime / 1000) / 1e12  # 转换为 TFLOP/s


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
