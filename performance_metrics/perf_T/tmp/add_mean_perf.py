import sys
import os
import json

sys.path.append('/home/lishangzhan/triton/torch_performance/GPU_efficiency/output_DeepSeek-R1_rag')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from add_mean import add_mean
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('add_mean', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        dtype = self.dtype if self.dtype is not None else torch.float32
        for i in range(12, 28):
            size = 2 ** i
            input_tensor = torch.rand(size, dtype=dtype)
            other_tensor = torch.rand(size, dtype=dtype)
            self.input_tensors.append((input_tensor, other_tensor))

    def to_cuda(self, input_tensor_tuple):
        input_cuda = input_tensor_tuple[0].cuda()
        other_cuda = input_tensor_tuple[1].cuda()
        return (input_cuda, other_cuda)
    
    def call_op(self, input_tensor_tuple):
        input, other = input_tensor_tuple
        return add_mean(input, other, dim=None)
    
    def get_gbps(self, input_tensor_tuple, runtime):
        input, other = input_tensor_tuple
        total_bytes = (input.numel() + other.numel()) * input.element_size() * 2
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor_tuple, runtime):
        input, other = input_tensor_tuple
        N = input.numel()
        flops = 3 * N  # 2*N (alpha*other + input) + N (sum)
        TFLOPS = flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
