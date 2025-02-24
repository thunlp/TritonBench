import sys
import os
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.attention_llama import triton_fa
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('attention_llama', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 12):  # Adjust the range as needed for your testing
            batch_size = 2 ** i
            num_heads = 8
            seq_len = 128
            head_dim = 64
            q = torch.rand((batch_size, num_heads, seq_len, head_dim), dtype=torch.float16)
            k = torch.rand((batch_size, num_heads, seq_len, head_dim), dtype=torch.float16)
            v = torch.rand((batch_size, num_heads, seq_len, head_dim), dtype=torch.float16)
            sm_scale = 1.0 / math.sqrt(head_dim)
            is_causal = False
            start_position = 0
            self.input_tensors.append((q, k, v, sm_scale, is_causal, start_position))

    def to_cuda(self, input_tensor):
        q, k, v, sm_scale, is_causal, start_position = input_tensor
        return (q.cuda(), k.cuda(), v.cuda(), sm_scale, is_causal, start_position)

    def call_op(self, input_tensor):
        q, k, v, sm_scale, is_causal, start_position = input_tensor
        return triton_fa(q, k, v, sm_scale, is_causal, start_position)

    def get_gbps(self, input_tensor, runtime):
        q, k, v, _, _, _ = input_tensor
        total_bytes = (q.numel() + k.numel() + v.numel() + q.numel()) * q.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        q, _, _, _, _, _ = input_tensor
        batch_size, num_heads, seq_len, head_dim = q.size()
        FLOPS = 2 * batch_size * num_heads * seq_len * head_dim * seq_len
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
