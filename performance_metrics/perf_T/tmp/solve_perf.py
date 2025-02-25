import sys
import os
import json

sys.path.append('/home/lishangzhan/triton/torch_performance/GPU_efficiency/output_DeepSeek-R1_rag')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from solve import solve
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('solve', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        # 生成不同规模的测试用例（n从2^4到2^11）
        for i in range(4, 12):
            n = 2 ** i
            # 生成可逆矩阵A（对角线增强确保非奇异）
            A = torch.randn(n, n, dtype=self.dtype) + n * torch.eye(n, dtype=self.dtype)
            # 生成右侧矩阵B（这里假设为单列向量）
            B = torch.randn(n, 1, dtype=self.dtype)
            self.input_tensors.append((A, B))
    
    def to_cuda(self, input_tensor):
        A, B = input_tensor
        return (A.cuda(), B.cuda())
        
    def call_op(self, input_tensor):
        A, B = input_tensor
        return solve(A, B)

    def get_gbps(self, input_tensor, runtime):
        A, B = input_tensor
        X_shape = B.shape  # 解X的形状与B相同
        # 总数据量 = 输入A大小 + 输入B大小 + 输出X大小
        total_bytes = (A.numel() + B.numel() + X_shape[0]*X_shape[1]) * A.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9  # 转换为GB/s
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        A, B = input_tensor
        n = A.shape[0]
        k = B.shape[1] if len(B.shape) > 1 else 1
        # 理论FLOPS估算：LU分解(2/3n³) + 解方程(2n²k)
        flops = (2/3) * (n**3) + 2 * (n**2) * k
        TFLOPS = flops / (runtime / 1000) / 1e12  # 转换为TFLOPS
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
