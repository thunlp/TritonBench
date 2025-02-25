import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.fused_cross_entropy_log_softmax import fused_cross_entropy_log_softmax
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_cross_entropy_log_softmax', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        num_classes = 128
        for i in range(12, 24):
            batch_size = 2 ** i
            input_tensor = torch.randn(batch_size, num_classes, dtype=torch.float32)
            target = torch.randint(0, num_classes, (batch_size,), dtype=torch.int64)
            self.input_tensors.append((input_tensor, target))

    def to_cuda(self, input_tensor):
        input_cuda = input_tensor[0].cuda()
        target_cuda = input_tensor[1].cuda()
        return (input_cuda, target_cuda)
    
    def call_op(self, input_tensor):
        input, target = input_tensor
        return fused_cross_entropy_log_softmax(input, target)
    
    def get_gbps(self, input_tensor, runtime):
        input, target = input_tensor
        input_bytes = input.numel() * input.element_size()
        target_bytes = target.numel() * target.element_size()
        output_bytes = 4
        total_bytes = 3 * input_bytes + target_bytes + output_bytes
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        input, target = input_tensor
        batch_size, num_classes = input.shape
        flops = batch_size * num_classes * 3 + batch_size
        TFLOPS = flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
