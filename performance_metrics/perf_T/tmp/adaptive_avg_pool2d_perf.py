import sys
import os
import json

sys.path.append('/home/lishangzhan/triton/torch_performance/GPU_efficiency/output_DeepSeek-R1_rag')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from adaptive_avg_pool2d import adaptive_avg_pool2d
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('adaptive_avg_pool2d', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        # Generate 4D tensors with increasing spatial dimensions
        for exp in range(8, 16):  # Adjust range based on memory constraints
            H = 2 ** exp
            W = 2 ** exp
            input_tensor = torch.randn(1, 3, H, W, dtype=self.dtype)
            output_size = (H // 2, W // 2)
            output_numel = 1 * 3 * output_size[0] * output_size[1]
            self.input_tensors.append((input_tensor, output_size, output_numel))

    def to_cuda(self, input_tuple):
        input_tensor, output_size, output_numel = input_tuple
        return (input_tensor.cuda(), output_size, output_numel)
    
    def call_op(self, input_tuple):
        input_tensor, output_size, _ = input_tuple
        return adaptive_avg_pool2d(input_tensor, output_size)

    def get_gbps(self, input_tuple, runtime):
        input_tensor, _, output_numel = input_tuple
        input_bytes = input_tensor.numel() * input_tensor.element_size()
        output_bytes = output_numel * input_tensor.element_size()  # Same dtype as input
        total_bytes = input_bytes + output_bytes
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tuple, runtime):
        input_tensor, output_size, output_numel = input_tuple
        H_in = input_tensor.shape[2]
        W_in = input_tensor.shape[3]
        H_out, W_out = output_size
        
        # Calculate average kernel size per output element
        avg_kH = H_in / H_out
        avg_kW = W_in / W_out
        
        # FLOPs per output element (avg operations per element)
        flops_per_element = avg_kH * avg_kW  # 1 mul + (k-1) adds per element
        
        total_flops = output_numel * flops_per_element
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
