import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.bitwise_and_binomial import bitwise_and_binomial
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('bitwise_and_binomial', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):
            size = 2 ** i
            input_tensor = torch.randint(0, 2, (size,), dtype=torch.int32)
            other_tensor = torch.randint(0, 2, (size,), dtype=torch.int32)
            total_count = torch.tensor(10, dtype=torch.int32)
            probs = torch.rand(size, dtype=torch.float32) * 0.5 + 0.25
            self.input_tensors.append((input_tensor, other_tensor, total_count, probs))

    def to_cuda(self, input_tuple):
        input_tensor, other_tensor, total_count, probs = input_tuple
        return (
            input_tensor.cuda(),
            other_tensor.cuda(),
            total_count.cuda(),
            probs.cuda()
        )

    def call_op(self, input_tuple):
        input_tensor, other_tensor, total_count, probs = input_tuple
        return bitwise_and_binomial(input_tensor, other_tensor, total_count, probs=probs)

    def get_gbps(self, input_tuple, runtime):
        input_tensor, other_tensor, total_count, probs = input_tuple
        input_bytes = input_tensor.numel() * input_tensor.element_size()
        other_bytes = other_tensor.numel() * other_tensor.element_size()
        total_count_bytes = total_count.numel() * total_count.element_size()
        probs_bytes = probs.numel() * probs.element_size()
        output_bytes = input_tensor.numel() * 8
        
        total_bytes = input_bytes + other_bytes + total_count_bytes + probs_bytes + output_bytes
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tuple, runtime):
        input_tensor, other_tensor, total_count, _ = input_tuple
        numel = input_tensor.numel()
        operations_per_element = total_count.item()
        total_ops = numel * operations_per_element
        TFLOPS = total_ops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
