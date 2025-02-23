import sys
import os
import json

sys.path.append('/home/lishangzhan/triton/torch_performance/GPU_efficiency/output_DeepSeek-R1')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scaled_add_dot import scaled_add_dot
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('scaled_add_dot', dtype=dtype, is_backward=is_backward, **kwargs)
        self.alpha = 1.0  # 可根据需要调整alpha的值

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):
            size = 2 ** i
            y = torch.rand(size, dtype=self.dtype or torch.float32)
            x = torch.rand(size, dtype=self.dtype or torch.float32)
            self.input_tensors.append((y, x))

    def to_cuda(self, input_tensor):
        y, x = input_tensor
        return (y.cuda(), x.cuda())
    
    def call_op(self, input_tensor):
        y, x = input_tensor
        return scaled_add_dot(y, x, self.alpha)
    
    def get_gbps(self, input_tensor, runtime):
        y, x = input_tensor
        n = y.numel()
        bytes_per_element = y.element_size()
        total_bytes = 6 * n * bytes_per_element  # 3n (y += a*x) + n (dot)
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        y, x = input_tensor
        n = y.numel()
        flops = 4 * n  # 2n (y += a*x) + 2n (dot)
        TFLOPS = flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
