import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.leaky_relu_conv2d import leaky_relu_conv2d
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('leaky_relu_conv2d', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 12):
            size = 128 * i
            input_tensor = torch.rand(1, 3, size, size, dtype=torch.float32)
            weight_tensor = torch.rand(6, 3, 3, 3, dtype=torch.float32)
            bias_tensor = torch.rand(6, dtype=torch.float32)
            self.input_tensors.append((input_tensor, weight_tensor, bias_tensor))

    def to_cuda(self, input_tensor):
        input_tensor, weight_tensor, bias_tensor = input_tensor
        return input_tensor.cuda(), weight_tensor.cuda(), bias_tensor.cuda()

    def call_op(self, input_tensor):
        input_tensor, weight_tensor, bias_tensor = input_tensor
        return leaky_relu_conv2d(input_tensor, weight_tensor, bias_tensor)

    def get_gbps(self, input_tensor, runtime):
        input_tensor, weight_tensor, bias_tensor = input_tensor
        total_bytes = (input_tensor.numel() + weight_tensor.numel() + bias_tensor.numel()) * input_tensor.element_size() * 4
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tensor, runtime):
        input_tensor, weight_tensor, bias_tensor = input_tensor
        FLOPS = input_tensor.numel() * weight_tensor.numel()
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
