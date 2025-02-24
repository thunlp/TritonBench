import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.tensordot import tensordot
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('tensordot', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        # 生成不同大小的输入张量，每个测试用例包含两个张量和dims参数
        for i in range(12, 22):
            size = 2 ** i
            # 构造满足dims=2的输入张量形状
            a_shape = (size // 64, 8, 8)  # a的总元素数为size
            b_shape = (8, 8, size // 64)  # b的总元素数也为size
            a = torch.randn(a_shape, dtype=torch.float16)
            b = torch.randn(b_shape, dtype=torch.float16)
            dims = 2  # 收缩a的最后两个维度和b的前两个维度
            self.input_tensors.append((a, b, dims))

    def to_cuda(self, input_tuple):
        # 将输入张量转移到CUDA
        a, b, dims = input_tuple
        return (a.cuda(), b.cuda(), dims)

    def call_op(self, input_tuple):
        # 调用tensordot算子
        a, b, dims = input_tuple
        return tensordot(a, b, dims)

    def get_gbps(self, input_tuple, runtime):
        # 计算GBPS：总数据量（输入+输出） / 运行时间
        a, b, dims = input_tuple
        a_numel = a.numel()
        b_numel = b.numel()
        # 根据输入形状推断输出形状
        x = a.shape[0]
        w = b.shape[2]
        output_numel = x * w
        # 总字节数（输入和输出）
        total_bytes = (a_numel + b_numel + output_numel) * a.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tuple, runtime):
        # 计算TFLOPS：浮点运算次数 / 运行时间
        a, b, dims = input_tuple
        x = a.shape[0]
        y, z = a.shape[1], a.shape[2]
        w = b.shape[2]
        # 每个输出元素需要2*y*z次浮点运算（乘加）
        flops = 2 * x * w * y * z
        TFLOPS = flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
