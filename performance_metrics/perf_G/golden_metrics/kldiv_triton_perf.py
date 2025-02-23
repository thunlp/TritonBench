import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.kldiv_triton import kldiv_forward_triton
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('kldiv_triton', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 15):
            size = 2 ** i
            y_pred = torch.rand((size, size), dtype=torch.float32)
            y_true = torch.rand((size, size), dtype=torch.float32)
            self.input_tensors.append((y_pred, y_true))

    def to_cuda(self, input_tensor):
        y_pred, y_true = input_tensor
        return (y_pred.cuda(), y_true.cuda())

    def call_op(self, input_tensor):
        y_pred, y_true = input_tensor
        return kldiv_forward_triton(y_pred, y_true, log_target=False, reduction='batchmean')

    def get_gbps(self, input_tensor, runtime):
        y_pred, _ = input_tensor
        total_bytes = 2 * y_pred.numel() * y_pred.element_size()  # y_pred, y_true
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        y_pred, _ = input_tensor
        FLOPS = 2 * y_pred.numel()  # Assuming 2 operations per element
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
