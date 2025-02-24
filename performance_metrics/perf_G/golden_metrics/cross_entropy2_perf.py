import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.cross_entropy2 import cross_entropy_fwd, cross_entropy_bwd
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('cross_entropy2', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        for i in range(2, 20):  # Adjust the range as needed for your testing
            n_rows = 2 ** i
            n_cols = 1000
            logits = torch.rand((n_rows, n_cols), dtype=torch.float32)
            labels = torch.randint(0, n_cols, (n_rows,), dtype=torch.int32)
            self.input_tensors.append((logits, labels))

    def to_cuda(self, input_tensor):
        logits, labels = input_tensor
        return logits.cuda(), labels.cuda()

    def call_op(self, input_tensor):
        logits, labels = input_tensor
        smoothing = 0.1
        logit_scale = 1.0
        lse_square_scale = 0.0
        ignored_index = -1
        total_classes = logits.size(1)
        class_start_idx = 0
        BLOCK_SIZE = 128
        HAS_SMOOTHING = True
        SPLIT = False
        return cross_entropy_fwd(logits, labels, smoothing, logit_scale, lse_square_scale, ignored_index, total_classes, class_start_idx, BLOCK_SIZE, HAS_SMOOTHING, SPLIT)

    def get_gbps(self, input_tensor, runtime):
        logits, _ = input_tensor
        total_bytes = logits.numel() * logits.element_size() * 2  # Read and write
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        logits, _ = input_tensor
        FLOPS = 2 * logits.numel()  # Assuming 2 operations per element
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
