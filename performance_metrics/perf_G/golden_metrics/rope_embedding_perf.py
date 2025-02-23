import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.rope_embedding import _rope_embedding_forward_impl
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('rope_embedding', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 12):  # Adjust the range based on expected tensor sizes
            batch_size = 2 ** i
            seq_len = 128  # Example sequence length
            n_heads = 8  # Example number of heads
            head_dim = 64  # Example head dimension
            Q = torch.rand(batch_size, n_heads, seq_len, head_dim, dtype=torch.float32)
            cos = torch.rand(seq_len, head_dim // 2, dtype=torch.float32)
            sin = torch.rand(seq_len, head_dim // 2, dtype=torch.float32)
            self.input_tensors.append((Q, cos, sin))

    def to_cuda(self, input_tensor):
        Q, cos, sin = input_tensor
        return Q.cuda(), cos.cuda(), sin.cuda()

    def call_op(self, input_tensor):
        Q, cos, sin = input_tensor
        return _rope_embedding_forward_impl(Q, cos, sin)

    def get_gbps(self, input_tensor, runtime):
        Q, _, _ = input_tensor
        total_bytes = 3 * Q.numel() * Q.element_size()  # Assuming Q, cos, sin are all involved
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        Q, _, _ = input_tensor
        FLOPS = 2 * Q.numel()  # Assuming each element involves 2 FLOP operations
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
