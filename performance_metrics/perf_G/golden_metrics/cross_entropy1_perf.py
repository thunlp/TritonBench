import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.cross_entropy1 import cross_entropy_loss
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('cross_entropy1', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        for i in range(2, 20):  # Adjust the range based on your testing needs
            n_rows = 2 ** i
            n_cols = 1000  # Fixed number of classes
            logits = torch.rand((n_rows, n_cols), dtype=torch.float16)
            labels = torch.randint(0, n_cols, (n_rows,), dtype=torch.int64)
            self.input_tensors.append((logits, labels))

    def to_cuda(self, input_tensor):
        logits, labels = input_tensor
        return logits.cuda(), labels.cuda()

    def call_op(self, input_tensor):
        logits, labels = input_tensor
        return cross_entropy_loss(logits, labels, label_smoothing=0.1)

    def get_gbps(self, input_tensor, runtime):
        logits, _ = input_tensor
        total_bytes = logits.numel() * logits.element_size() * 2  # Read and write
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        logits, _ = input_tensor
        # Assuming each element involves a few operations, e.g., exp, log, etc.
        FLOPS = 5 * logits.numel()  # Rough estimate
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
