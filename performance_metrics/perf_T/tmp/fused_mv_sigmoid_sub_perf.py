import sys
import os
import json

sys.path.append('/home/lishangzhan/triton/torch_performance/GPU_efficiency/output_DeepSeek-R1')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fused_mv_sigmoid_sub import fused_mv_sigmoid_sub
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_mv_sigmoid_sub', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        # 测试不同尺寸的输入矩阵（方阵）和向量
        for k in range(6, 16):  # 调整范围以避免内存不足
            n = 2 ** k
            input_matrix = torch.rand(n, n, dtype=torch.float32)
            vec = torch.rand(n, dtype=torch.float32)
            other = 0.5  # 使用标量进行测试
            self.input_tensors.append((input_matrix, vec, other))

    def to_cuda(self, input_tuple):
        input_matrix, vec, other = input_tuple
        # 将张量转移到CUDA，标量保持不变
        return (input_matrix.cuda(), vec.cuda(), other)

    def call_op(self, input_tuple):
        input_matrix, vec, other = input_tuple
        # 调用算子，alpha使用默认值1
        return fused_mv_sigmoid_sub(input_matrix, vec, other, alpha=1)

    def get_gbps(self, input_tuple, runtime):
        input_matrix, vec, other = input_tuple
        n, m = input_matrix.shape
        # 总数据量 = 输入矩阵(n*m) + 向量(m) + 输出向量(n)
        total_bytes = (n * m + m + n * 5) * input_matrix.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tuple, runtime):
        input_matrix, vec, other = input_tuple
        n, m = input_matrix.shape
        # FLOPs计算:
        # MV: 2*n*m (矩阵乘法)
        # Sigmoid: 4*n (假设每个元素4次操作)
        # Sub: 1*n (减法)
        total_flops = 2 * n * m + 5 * n
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
