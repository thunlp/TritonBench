import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.grid_sample_with_affine import grid_sample_with_affine
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('grid_sample_with_affine', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        H_in = 256  # 固定输入尺寸
        for i in range(5, 11):  # 测试不同输出尺寸：32x32 到 1024x1024
            H_out = 2 ** i
            size = (1, 3, H_out, H_out)
            input_tensor = torch.randn(1, 3, H_in, H_in, dtype=self.dtype)
            theta = torch.randn(1, 2, 3, dtype=self.dtype)
            self.input_tensors.append((input_tensor, theta, size))

    def to_cuda(self, input_tuple):
        input_tensor, theta, size = input_tuple
        return (input_tensor.cuda(), theta.cuda(), size)

    def call_op(self, input_tuple):
        input_tensor, theta, size = input_tuple
        return grid_sample_with_affine(input_tensor, theta, size)

    def get_gbps(self, input_tuple, runtime):
        input_tensor, _, size = input_tuple
        N, C, H_out, W_out = size
        input_el = input_tensor.numel()
        output_el = N * C * H_out * W_out
        element_size = input_tensor.element_size()
        total_bytes = (input_el + output_el) * element_size
        return total_bytes / (runtime / 1000) / 1e9

    def get_tflops(self, input_tuple, runtime):
        _, _, size = input_tuple
        N, C, H_out, W_out = size
        # 仿射变换计算量 (每个网格点 10 FLOP)
        affine_flops = N * H_out * W_out * 10
        # 网格采样计算量 (每个输出元素 7 FLOP)
        grid_sample_flops = N * C * H_out * W_out * 7
        total_flops = affine_flops + grid_sample_flops
        return total_flops / (runtime / 1000) / 1e12

# 测试 float32 性能
pm = performance_metrics(dtype=torch.float32)
pm.run_benchmark()

# 测试 float16 性能 
pm_fp16 = performance_metrics(dtype=torch.float16)
pm_fp16.run_benchmark()


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
