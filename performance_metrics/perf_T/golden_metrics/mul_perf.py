import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.mul import mul
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('mul', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):
            size = 2 ** i
            input_tensor = torch.rand(size, dtype=torch.float16)
            other_tensor = torch.rand(size, dtype=torch.float16)
            self.input_tensors.append((input_tensor, other_tensor))

    def to_cuda(self, input_tensor_tuple):
        return (input_tensor_tuple[0].cuda(), input_tensor_tuple[1].cuda())
    
    def call_op(self, input_tensor_tuple):
        return mul(input_tensor_tuple[0], input_tensor_tuple[1])
    
    def get_gbps(self, input_tensor_tuple, runtime):
        input1, input2 = input_tensor_tuple
        element_size = input1.element_size()
        total_bytes = (input1.numel() + input2.numel() + input1.numel()) * element_size
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor_tuple, runtime):
        flops = input_tensor_tuple[0].numel()  # 1 FLOP per element-wise multiplication
        TFLOPS = flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
