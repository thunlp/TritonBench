import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.gelu_conv2d import gelu_conv2d
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('gelu_conv2d', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        in_channels = 16
        out_channels = 32
        kernel_size = 3
        for i in range(12, 26):
            size = 2 ** i
            h = int((size / (1 * in_channels)) ** 0.5)
            while h * h * 1 * in_channels > size and h > 0:
                h -= 1
            w = h
            if h == 0:
                continue
            input_tensor = torch.randn(16, in_channels, h, w, dtype=torch.float32)
            weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.float32)
            self.input_tensors.append((input_tensor, weight))
    
    def to_cuda(self, input_tuple):
        input_tensor, weight_tensor = input_tuple
        return (input_tensor.cuda(), weight_tensor.cuda())
        
    def call_op(self, input_tuple):
        input_tensor, weight_tensor = input_tuple
        return gelu_conv2d(input_tensor, weight_tensor, bias=None, stride=1, padding=1, dilation=1, groups=1, approximate='none')
    
    def get_gbps(self, input_tuple, runtime):
        input_tensor, weight_tensor = input_tuple
        N, C_in, H, W = input_tensor.shape
        C_out = weight_tensor.shape[0]
        output_numel = N * C_out * H * W
        input_bytes = input_tensor.numel() * input_tensor.element_size()
        weight_bytes = weight_tensor.numel() * weight_tensor.element_size()
        output_bytes = output_numel * input_tensor.element_size()
        total_bytes = input_bytes + weight_bytes + output_bytes * 3
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tuple, runtime):
        input_tensor, weight_tensor = input_tuple
        N, C_in, H, W = input_tensor.shape
        C_out, _, K, _ = weight_tensor.shape
        conv_flops = N * C_out * H * W * 2 * C_in * K * K 
        gelu_flops = 4 * N * C_out * H * W 
        total_flops = conv_flops + gelu_flops
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
