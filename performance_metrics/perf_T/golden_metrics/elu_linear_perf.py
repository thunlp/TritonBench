import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.elu_linear import elu_linear
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('elu_linear', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(8, 17):
            batch_size = 1024
            in_features = 2 ** i
            out_features = 256
            input_tensor = torch.randn(batch_size, in_features, dtype=self.dtype)
            weight = torch.randn(out_features, in_features, dtype=self.dtype)
            bias = torch.randn(out_features, dtype=self.dtype)
            self.input_tensors.append((input_tensor, weight, bias))

    def to_cuda(self, input_tuple):
        input_tensor, weight, bias = input_tuple
        return (input_tensor.cuda(), weight.cuda(), bias.cuda())
    
    def call_op(self, input_tuple):
        input_tensor, weight, bias = input_tuple
        return elu_linear(input_tensor, weight, bias)

    def get_gbps(self, input_tuple, runtime):
        input_tensor, weight, bias = input_tuple
        batch_size, in_features = input_tensor.shape
        out_features = weight.shape[0]
        element_size = input_tensor.element_size()
        
        total_bytes = (input_tensor.numel() + weight.numel() + bias.numel() + 
                       batch_size * out_features) * element_size * 2
        return total_bytes / (runtime / 1000) / 1e9

    def get_tflops(self, input_tuple, runtime):
        input_tensor, weight, bias = input_tuple
        batch_size, in_features = input_tensor.shape
        out_features = weight.shape[0]
        
        flops_linear = 2 * batch_size * in_features * out_features
        flops_bias = batch_size * out_features
        flops_elu = 3 * batch_size * out_features
        
        total_flops = flops_linear + flops_bias + flops_elu
        return total_flops / (runtime / 1000) / 1e12


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
