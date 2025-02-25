import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.fused_avg_pool2d_cosine_similarity import fused_avg_pool2d_cosine_similarity
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, kernel_size=3, stride=2, padding=1, eps=1e-8, **kwargs):
        super().__init__('fused_avg_pool2d_cosine_similarity', dtype=dtype, is_backward=is_backward, **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.eps = eps

    def get_input_tensors(self):
        self.input_tensors = []
        for size in range(4, 11):
            print(size)
            spatial_size = 2 ** size
            batch_size = 16
            channels = 64
            H = W = spatial_size
            x1 = torch.randn(batch_size, channels, H, W, dtype=self.dtype)
            x2 = torch.randn_like(x1)
            self.input_tensors.append((x1, x2))
    
    def to_cuda(self, input_tensor):
        x1, x2 = input_tensor
        return (x1.cuda(), x2.cuda())
    
    def call_op(self, input_tensor):
        x1, x2 = input_tensor
        return fused_avg_pool2d_cosine_similarity(
            x1, x2,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            eps=self.eps
        )
    
    def get_gbps(self, input_tensor, runtime):
        x1, x2 = input_tensor
        H = x1.size(2)
        W = x1.size(3)
        H_out = (H + 2*self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2*self.padding - self.kernel_size) // self.stride + 1
        
        input_bytes = (x1.numel() + x2.numel()) * x1.element_size()
        output_bytes = 1 * 1 * H_out * W_out * x1.element_size()
        total_bytes = input_bytes + output_bytes
        
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        x1, x2 = input_tensor
        C = x1.size(1)
        H, W = x1.size()[2:]
        
        H_out = (H + 2*self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2*self.padding - self.kernel_size) // self.stride + 1
        
        flops_cosine = H * W * 6 * C
        
        flops_pool = H_out * W_out * (self.kernel_size**2)
        
        total_flops = flops_cosine + flops_pool
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
