import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.solve_symmetric_ldl import solve_symmetric_ldl
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('solve_symmetric_ldl', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        # 生成不同尺寸的对称正定矩阵和向量
        for i in range(3, 10):  # 测试规模：256x256到8192x8192
            n = 2 ** i
            # 生成对称正定矩阵A
            X = torch.rand(n, n, dtype=torch.float32)
            A = X @ X.T  # 确保对称
            A += torch.eye(n, dtype=torch.float32) * 1e-6  # 确保正定性
            # 生成右侧向量b
            b = torch.rand(n, 1, dtype=torch.float32)
            self.input_tensors.append((A, b))

    def to_cuda(self, input_tensor):
        # 将输入元组中的每个张量移动到CUDA
        A, b = input_tensor
        return (A.cuda(), b.cuda())
    
    def call_op(self, input_tensor):
        # 解算线性方程组
        A, b = input_tensor
        return solve_symmetric_ldl(A, b)
    
    def get_gbps(self, input_tensor, runtime):
        # 计算内存吞吐量（GB/s）
        A, b = input_tensor
        input_bytes = (A.numel() + b.numel()) * A.element_size()  # 输入数据量
        output_bytes = b.numel() * b.element_size()               # 输出数据量
        total_bytes = (input_bytes + output_bytes)                # 总数据量
        GBPS = total_bytes / (runtime / 1000) / 1e9               # 转换GB单位
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        # 估算计算吞吐量（TFLOP/s）
        A, b = input_tensor
        n = A.size(0)
        # LDL分解复杂度约(1/3)n³，解方程约2n²次运算
        flops = (n**3 / 3) + 2 * n**2
        TFLOPS = flops / (runtime / 1000) / 1e12  # 转换为Tera单位
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
