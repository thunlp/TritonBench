import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.fused_lu_solve import fused_lu_solve  # 正确引入算子
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_lu_solve', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        # 生成不同规模的测试数据（A为方阵，b为向量）
        for exp in range(8, 13):  # 测试矩阵维度从 2^8=256 到 2^12=4096
            n = 2 ** exp
            A = torch.rand(n, n, dtype=torch.float32)
            b = torch.rand(n, dtype=torch.float32)
            self.input_tensors.append((A, b))  # 存储为元组格式

    def to_cuda(self, input_tensor):
        # 将CPU张量转移到CUDA（元组解包后逐个转移）
        A, b = input_tensor
        return (A.cuda(), b.cuda())
    
    def call_op(self, input_tensor):
        # 算子调用（解包CUDA张量）
        A, b = input_tensor
        return fused_lu_solve(A, b)
    
    def get_gbps(self, input_tensor, runtime):
        # 计算GBPS（总数据量/运行时间）
        A, b = input_tensor
        n = A.shape[0]
        element_size = A.element_size()  # 获取数据类型大小（e.g. float32=4）
        
        # 总数据量 = 输入矩阵A + 输入向量b + 输出向量x
        total_bytes = (A.numel() + b.numel() + n) * element_size
        
        # 单位转换：bytes/(ms/1000) -> GB/s
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        # 计算TFLOPS（浮点运算量/运行时间）
        A, b = input_tensor
        n = A.shape[0]
        
        # LU分解计算量约 (2/3)n^3 FLOPS，求解三角方程组约 2n^2 FLOPS
        flops = (2/3) * n**3 + 2 * n**2
        
        # 单位转换：FLOPS/(ms/1000) -> TFLOPS
        TFLOPS = flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
