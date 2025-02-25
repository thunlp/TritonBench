import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.fused_cross_entropy_softmax_layernorm import fused_cross_entropy_softmax_layernorm
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_cross_entropy_softmax_layernorm', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(8, 20):
            batch_size = 2 ** i
            vocab_size = 1024
            logits = torch.randn(batch_size, vocab_size, dtype=torch.float32)
            targets = torch.randint(0, vocab_size, (batch_size,), dtype=torch.int64)
            normalized_shape = vocab_size
            self.input_tensors.append((logits, targets, normalized_shape))

    def to_cuda(self, input_tuple):
        logits, targets, normalized_shape = input_tuple
        return (logits.cuda(), targets.cuda(), normalized_shape)
    
    def call_op(self, input_tuple):
        logits, targets, normalized_shape = input_tuple
        return fused_cross_entropy_softmax_layernorm(logits, targets, normalized_shape)
    
    def get_gbps(self, input_tensor, runtime):
        logits, _, _ = input_tensor
        total_bytes = logits.numel() * logits.element_size() * 6
        return total_bytes / (runtime / 1000) / 1e9
    
    def get_tflops(self, input_tensor, runtime):
        logits, _, _ = input_tensor
        # softmax: 2n (exp + sum)
        # cross_entropy: n (log_softmax) + n (gather)
        # layer_norm: 3n (mean + var + norm)
        total_flops = logits.numel() * 6
        return total_flops / (runtime / 1000) / 1e12

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
