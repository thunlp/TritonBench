import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.batch_norm import batch_norm
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('batch_norm', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for k in range(5, 13):
            print(k)
            H = 128 * k
            W = 128
            C = 32
            N = 32
            input_tensor = torch.randn(N, C, H, W, dtype=torch.float16)
            running_mean = torch.randn(C)
            running_var = torch.abs(torch.randn(C)) + 1e-5
            weight = torch.randn(C)
            bias = torch.randn(C)
            self.input_tensors.append((input_tensor, running_mean, running_var, weight, bias))

    def to_cuda(self, input_tuple):
        input_tensor, running_mean, running_var, weight, bias = input_tuple
        return (
            input_tensor.cuda(),
            running_mean.cuda(),
            running_var.cuda(),
            weight.cuda(),
            bias.cuda()
        )

    def call_op(self, input_tuple):
        input_tensor, running_mean, running_var, weight, bias = input_tuple
        return batch_norm(input_tensor, running_mean, running_var, weight, bias, 
                          training=False, momentum=0.1, eps=1e-5)

    def get_gbps(self, input_tuple, runtime):
        input_tensor = input_tuple[0]
        total_bytes = input_tensor.numel() * input_tensor.element_size() * 2
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tuple, runtime):
        input_tensor = input_tuple[0]
        flops = input_tensor.numel() * 4
        TFLOPS = flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
