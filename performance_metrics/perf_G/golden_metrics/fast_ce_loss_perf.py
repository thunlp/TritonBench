import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.fast_ce_loss import fast_cross_entropy_loss
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fast_ce_loss', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 16):  # Adjust the range for different sizes
            batch_size = 2 ** i
            seq_len = 128  # Fixed sequence length
            vocab_size = 1000  # Fixed vocabulary size
            logits = torch.rand((batch_size, seq_len, vocab_size), dtype=torch.float16)
            labels = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.int64)
            self.input_tensors.append((logits, labels))

    def to_cuda(self, input_tensor):
        logits, labels = input_tensor
        return logits.cuda(), labels.cuda()

    def call_op(self, input_tensor):
        logits, labels = input_tensor
        return fast_cross_entropy_loss(logits, labels)

    def get_gbps(self, input_tensor, runtime):
        logits, labels = input_tensor
        total_bytes = logits.numel() * logits.element_size() + labels.numel() * labels.element_size()  # Read and write
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        logits, _ = input_tensor
        # Assuming each element involves a few operations, adjust as necessary
        FLOPS = 2 * logits.numel()  # Example: 2 operations per element
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
