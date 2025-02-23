import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 引入matmul算子
from TritonBench_v1.streamk_matmul import matmul
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('streamk_matmul', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        # 定义不同尺寸的输入张量
        for i in range(2, 33):  # 2^12 到 2^28 大小
            M = N = K = 128 * i
            # 定义矩阵A和B，随机初始化
            input_tensor = (torch.rand(M, K, dtype=torch.float16), torch.rand(K, N, dtype=torch.float16))
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        # 将输入张量转移到CUDA设备
        return (input_tensor[0].cuda(), input_tensor[1].cuda())

    def call_op(self, input_tensor):
        # 调用matmul算子进行矩阵乘法
        a, b = input_tensor
        return matmul.forward(None, a, b, grid=1)  # 假设只使用一个grid

    def get_gbps(self, input_tensor, runtime):
        # 计算GBPS
        a, b = input_tensor
        total_bytes = (a.numel() + b.numel() + a.size(0) * b.size(1)) * a.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        # 计算TFLOPS
        a, b = input_tensor
        FLOPS = 2 * float(a.size(0)) * float(a.size(1)) * float(b.size(1))
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
