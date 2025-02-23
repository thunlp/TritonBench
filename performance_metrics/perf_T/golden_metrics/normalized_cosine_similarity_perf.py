import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.normalized_cosine_similarity import normalized_cosine_similarity
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('normalized_cosine_similarity', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(6, 16):  # 测试不同规模的数据 (2^12 到 2^27)
            size = 2 ** i
            # 生成两个相同形状的输入张量 (N, 1024)
            x1 = torch.rand((size, 1024), dtype=self.dtype)
            x2 = torch.rand((size, 1024), dtype=self.dtype)
            self.input_tensors.append((x1, x2))

    def to_cuda(self, input_tensor):
        # 将CPU张量迁移到GPU
        x1, x2 = input_tensor
        return (x1.cuda(), x2.cuda())
    
    def call_op(self, input_tensor):
        # 调用目标算子
        x1, x2 = input_tensor
        return normalized_cosine_similarity(x1, x2, dim=1)
    
    def get_gbps(self, input_tensor, runtime):
        # 计算内存带宽 (GB/s)
        x1, x2 = input_tensor
        input_bytes = (x1.numel() + x2.numel()) * x1.element_size()  # 输入数据量
        output_bytes = x1.shape[0] * x1.element_size()               # 输出数据量
        total_bytes = input_bytes * 3 + output_bytes                     # 总数据量
        return total_bytes / (runtime / 1000) / 1e9                  # 转换为GB/s
    
    def get_tflops(self, input_tensor, runtime):
        # 计算计算吞吐量 (TFLOPS)
        x1, x2 = input_tensor
        N, D = x1.shape
        flops = 8 * N * D  # 总浮点运算量估算 (每个元素约8次运算)
        return flops / (runtime / 1000) / 1e12  # 转换为TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
