import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.kv_cache_filling import fill_kv_cache
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('kv_cache_filling', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 18):
            batch_size = 16
            num_heads = 8
            seq_len = 2 ** i
            head_dim = 32
            head_dim_v = 32
            block_size = 8

            k_states = torch.rand((batch_size, seq_len, num_heads, head_dim), dtype=torch.float16)
            v_states = torch.rand((batch_size, seq_len, num_heads, head_dim_v), dtype=torch.float16)
            k_caches = torch.zeros((batch_size, block_size, num_heads, head_dim), dtype=torch.float16)
            v_caches = torch.zeros((batch_size, block_size, num_heads, head_dim_v), dtype=torch.float16)
            q_start_loc = torch.randint(0, seq_len, (batch_size,), dtype=torch.int32)
            q_seq_length = torch.randint(1, seq_len, (batch_size,), dtype=torch.int32)
            kv_seq_length = q_seq_length + torch.randint(0, seq_len // 2, (batch_size,), dtype=torch.int32)
            block_offsets = torch.randint(0, block_size, (batch_size, block_size), dtype=torch.int32)

            self.input_tensors.append((k_states, v_states, k_caches, v_caches, q_start_loc, q_seq_length, kv_seq_length, block_offsets))

    def to_cuda(self, input_tensor):
        return tuple(tensor.cuda() for tensor in input_tensor)

    def call_op(self, input_tensor):
        k_states, v_states, k_caches, v_caches, q_start_loc, q_seq_length, kv_seq_length, block_offsets = input_tensor
        fill_kv_cache(
            k_states,
            v_states,
            k_caches,
            v_caches,
            q_start_loc,
            q_seq_length,
            kv_seq_length,
            max_q_seq_length=k_states.size(1),
            block_offsets=block_offsets
        )
        return k_caches, v_caches

    def get_gbps(self, input_tensor, runtime):
        k_states, v_states, k_caches, v_caches, _, _, _, _ = input_tensor
        total_bytes = (
            k_states.numel() * k_states.element_size() +
            v_states.numel() * v_states.element_size() +
            k_caches.numel() * k_caches.element_size() +
            v_caches.numel() * v_caches.element_size()
        )
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tensor, runtime):
        k_states, _, _, _, _, _, _, _ = input_tensor
        batch_size, seq_len, num_heads, head_dim = k_states.size()
        FLOPS = 2 * batch_size * seq_len * num_heads * head_dim
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
