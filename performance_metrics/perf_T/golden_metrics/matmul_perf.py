import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.matmul import matmul  # 正确引入矩阵乘法算子
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('matmul', dtype=dtype, is_backward=is_backward, **kwargs)  # 修正算子名称

    def get_input_tensors(self):
        self.input_tensors = []
        # 生成不同尺寸的矩阵对 (M, K) x (K, N)
        for i in range(2, 33):  # 从 2^8=256 到 2^14=16384
            size = 128 * i
            # 生成可相乘的矩阵对 (size, size) x (size, size)
            tensor1 = torch.rand(size, size, dtype=torch.float16)
            tensor2 = torch.rand(size, size, dtype=torch.float16)
            self.input_tensors.append((tensor1, tensor2))

    def to_cuda(self, input_tensor):
        # 将两个输入张量分别移动到 CUDA
        tensor1, tensor2 = input_tensor
        return (tensor1.cuda(), tensor2.cuda())
    
    def call_op(self, input_tensor):
        # 调用矩阵乘法算子
        tensor1, tensor2 = input_tensor
        return matmul(tensor1, tensor2)
    
    def get_gbps(self, input_tensor, runtime):
        # 计算总数据量 (输入+输出)
        tensor1, tensor2 = input_tensor
        M, K = tensor1.shape
        _, N = tensor2.shape
        element_size = tensor1.element_size()
        
        # 输入输出总字节数 = (M*K + K*N)输入 + M*N输出
        total_bytes = (M*K + K*N + M*N) * element_size
        
        # 转换为 GB/s（注意 runtime 是毫秒）
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        # 计算浮点运算次数 (2*M*N*K)
        tensor1, tensor2 = input_tensor
        M, K = tensor1.shape
        _, N = tensor2.shape
        
        flops = 2 * M * N * K  # 矩阵乘法运算次数公式
        TFLOPS = flops / (runtime / 1000) / 1e12  # 转换为 TFLOP/s
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
