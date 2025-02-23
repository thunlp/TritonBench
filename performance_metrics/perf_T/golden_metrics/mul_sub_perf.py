import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.mul_sub import mul_sub
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, alpha=1, **kwargs):
        super().__init__('mul_sub', dtype=dtype, is_backward=is_backward, **kwargs)
        self.alpha = alpha  # 支持自定义alpha参数

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):  # 从2^12到2^27的不同size
            size = 2 ** i
            # 生成三个相同shape的输入张量（input, other_mul, other_sub）
            input_tensor = torch.rand(size, dtype=self.dtype)
            other_mul = torch.rand(size, dtype=self.dtype)
            other_sub = torch.rand(size, dtype=self.dtype)
            self.input_tensors.append((input_tensor, other_mul, other_sub))

    def to_cuda(self, input_tuple):
        # 将元组中的每个张量转移到CUDA
        input_tensor, other_mul, other_sub = input_tuple
        return (
            input_tensor.cuda(),
            other_mul.cuda(),
            other_sub.cuda()
        )
    
    def call_op(self, input_tuple):
        # 解包输入元组并调用算子
        input_tensor, other_mul, other_sub = input_tuple
        return mul_sub(
            input_tensor,
            other_mul,
            other_sub,
            alpha=self.alpha  # 传递alpha参数
        )
    
    def get_gbps(self, input_tuple, runtime):
        # 计算总内存吞吐量（GB/s）
        input_tensor, other_mul, other_sub = input_tuple
        element_size = input_tensor.element_size()
        # 输入输出总字节数 = 3输入 + 1输出
        total_bytes = (input_tensor.numel() * 3 + input_tensor.numel()) * element_size
        GBPS = total_bytes / (runtime / 1000) / 1e9  # 转换为GB/s
        return GBPS
    
    def get_tflops(self, input_tuple, runtime):
        # 计算计算吞吐量（TFLOPS）
        input_tensor = input_tuple[0]
        num_elements = input_tensor.numel()
        # 每个元素执行2次乘法和1次减法（共3次操作）
        flops = 3 * num_elements
        TFLOPS = flops / (runtime / 1000) / 1e12  # 转换为TFLOPS
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
