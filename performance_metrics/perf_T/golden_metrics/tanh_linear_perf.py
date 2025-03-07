import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.tanh_linear import tanh_linear
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('tanh_linear', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(5, 16):
            print(i)
            in_features = 2 ** i
            out_features = 2 ** (i - 2)
            batch_size = 2 ** (i - 3)
            input_tensor = torch.randn(batch_size, in_features, dtype=self.dtype)
            weight = torch.randn(out_features, in_features, dtype=self.dtype)
            bias = torch.randn(out_features, dtype=self.dtype)
            self.input_tensors.append((input_tensor, weight, bias))

    def to_cuda(self, input_tuple):
        input_tensor, weight, bias = input_tuple
        input_cuda = input_tensor.cuda()
        weight_cuda = weight.cuda()
        bias_cuda = bias.cuda() if bias is not None else None
        return (input_cuda, weight_cuda, bias_cuda)

    def call_op(self, input_tuple):
        input_tensor, weight, bias = input_tuple
        return tanh_linear(input_tensor, weight, bias)

    def get_gbps(self, input_tuple, runtime):
        input_tensor, weight, bias = input_tuple
        B, I = input_tensor.shape
        O, _ = weight.shape
        
        total_bytes = (B * I + O * I + O + B * O) * input_tensor.element_size()
        return total_bytes / (runtime / 1000) / 1e9

    def get_tflops(self, input_tuple, runtime):
        input_tensor, weight, bias = input_tuple
        B, I = input_tensor.shape
        O, _ = weight.shape
        
        flops = 2 * B * I * O + 2 * B * O
        return flops / (runtime / 1000) / 1e12


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
