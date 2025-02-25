import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.pixel_shuffle_conv2d import pixel_shuffle_conv2d
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('pixel_shuffle_conv2d', dtype=dtype, is_backward=is_backward, **kwargs)
        self.upscale_factor = 2
        self.kernel_size = 3
        self.in_channels = 3

    def get_input_tensors(self):
        self.input_tensors = []
        for spatial_exp in range(4, 11):
            H = W = 2 ** spatial_exp
            input_tensor = torch.randn(16, self.in_channels, H, W, dtype=torch.float32)
            
            out_channels = self.in_channels * (self.upscale_factor ** 2)
            weight = torch.randn(out_channels, self.in_channels, self.kernel_size, self.kernel_size, 
                                dtype=torch.float32)
            
            bias = torch.randn(out_channels, dtype=torch.float32)
            
            self.input_tensors.append((input_tensor, weight, bias))

    def to_cuda(self, input_tuple):
        input_tensor, weight, bias = input_tuple
        return (input_tensor.cuda(), weight.cuda(), bias.cuda())
    
    def call_op(self, input_tuple):
        input_tensor, weight, bias = input_tuple
        return pixel_shuffle_conv2d(
            input_tensor, 
            weight, 
            bias,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            upscale_factor=self.upscale_factor
        )
    
    def get_gbps(self, input_tuple, runtime):
        input_tensor, weight, _ = input_tuple
        N, C_in, H_in, W_in = input_tensor.shape
        
        H_conv = H_in - self.kernel_size + 1
        W_conv = W_in - self.kernel_size + 1
        
        H_out = H_conv * self.upscale_factor
        W_out = W_conv * self.upscale_factor
        C_out = weight.shape[0] // (self.upscale_factor ** 2)
        
        input_bytes = input_tensor.numel() * input_tensor.element_size()
        output_bytes = N * C_out * H_out * W_out * input_tensor.element_size()
        total_bytes = input_bytes * 3 + output_bytes
        
        return total_bytes / (runtime / 1000) / 1e9

    def get_tflops(self, input_tuple, runtime):
        input_tensor, weight, _ = input_tuple
        N, C_in, H_in, W_in = input_tensor.shape
        
        H_conv = H_in - self.kernel_size + 1
        W_conv = W_in - self.kernel_size + 1
        
        flops_per_output = C_in * (self.kernel_size ** 2) * 2
        total_flops = N * weight.shape[0] * H_conv * W_conv * flops_per_output
        
        return total_flops / (runtime / 1000) / 1e12


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
