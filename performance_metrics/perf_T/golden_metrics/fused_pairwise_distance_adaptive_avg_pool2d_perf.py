import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.fused_pairwise_distance_adaptive_avg_pool2d import fused_pairwise_distance_adaptive_avg_pool2d
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_pairwise_distance_adaptive_avg_pool2d', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 14):
            hw = 2 ** i
            x1 = torch.randn(16, 3, hw, hw, dtype=torch.float32)
            x2 = torch.randn(16, 3, hw, hw, dtype=torch.float32)
            output_size = (hw // 2, hw // 2)
            self.input_tensors.append((x1, x2, output_size))
    
    def to_cuda(self, input_tuple):
        x1, x2, output_size = input_tuple
        return (x1.cuda(), x2.cuda(), output_size)
        
    def call_op(self, input_tuple):
        x1, x2, output_size = input_tuple
        return fused_pairwise_distance_adaptive_avg_pool2d(x1, x2, output_size)
    
    def get_gbps(self, input_tuple, runtime):
        x1, x2, output_size = input_tuple
        x1_bytes = x1.numel() * x1.element_size()
        x2_bytes = x2.numel() * x2.element_size()
        B = x1.size(0)
        dist_bytes = B * x1.element_size()
        total_bytes = x1_bytes + x2_bytes + dist_bytes + x1_bytes / 4 * 6
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tuple, runtime):
        x1, x2, output_size = input_tuple
        B, C, H, W = x1.shape
        if isinstance(output_size, int):
            H_prime, W_prime = output_size, output_size
        else:
            H_prime, W_prime = output_size
        
        pool_flops = 2 * B * C * (H * W + H_prime * W_prime)
        diff_flops = B * C * H_prime * W_prime
        norm_flops = B * 2 * C * H_prime * W_prime
        total_flops = pool_flops + diff_flops + norm_flops
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
