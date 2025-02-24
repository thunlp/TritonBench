import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.context_attn_nopad import context_attention_fwd
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('context_attn_nopad', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(7, 16):  # Adjust the range as needed for different sizes
            batch_size = 2 ** i
            seq_len = 128  # Fixed sequence length for simplicity
            head_dim = 64  # Fixed head dimension
            num_heads = 8  # Number of attention heads

            q = torch.rand((batch_size, num_heads, seq_len, head_dim), dtype=torch.float16)
            k = torch.rand((batch_size, num_heads, seq_len, head_dim), dtype=torch.float16)
            v = torch.rand((batch_size, num_heads, seq_len, head_dim), dtype=torch.float16)
            o = torch.zeros_like(q)
            b_start_loc = torch.zeros(batch_size, dtype=torch.int32)
            b_seq_len = torch.full((batch_size,), seq_len, dtype=torch.int32)

            self.input_tensors.append((q, k, v, o, b_start_loc, b_seq_len, seq_len))

    def to_cuda(self, input_tensor):
        q, k, v, o, b_start_loc, b_seq_len, max_input_len = input_tensor
        return (q.cuda(), k.cuda(), v.cuda(), o.cuda(), b_start_loc.cuda(), b_seq_len.cuda(), max_input_len)

    def call_op(self, input_tensor):
        q, k, v, o, b_start_loc, b_seq_len, max_input_len = input_tensor
        context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_input_len)
        return o

    def get_gbps(self, input_tensor, runtime):
        q, k, v, o, b_start_loc, b_seq_len, max_input_len = input_tensor
        total_bytes = (q.numel() + k.numel() + v.numel() + o.numel()) * q.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        q, k, v, o, b_start_loc, b_seq_len, max_input_len = input_tensor
        batch_size, num_heads, seq_len, head_dim = q.shape
        FLOPS = 2 * batch_size * num_heads * seq_len * seq_len * head_dim  # Simplified FLOPS calculation
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
