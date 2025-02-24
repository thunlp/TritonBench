import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.kv_cache_copy import copy_kv_to_blocked_cache
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('kv_cache_copy', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 18):  # Adjust the range as needed for your testing
            bsz = 2 ** i
            num_kv_heads = 8  # Example value, adjust as needed
            head_dim = 64  # Example value, adjust as needed
            block_size = 128  # Example value, adjust as needed
            max_blocks_per_sequence = 10  # Example value, adjust as needed

            k = torch.rand(bsz, 1, num_kv_heads, head_dim, dtype=torch.float16)
            v = torch.rand(bsz, 1, num_kv_heads, head_dim, dtype=torch.float16)
            k_cache = torch.rand(max_blocks_per_sequence, num_kv_heads, block_size, head_dim, dtype=torch.float16)
            v_cache = torch.rand(max_blocks_per_sequence, num_kv_heads, block_size, head_dim, dtype=torch.float16)
            kv_lengths = torch.randint(1, block_size + 1, (bsz,), dtype=torch.int32)
            block_tables = torch.randint(0, max_blocks_per_sequence, (bsz, max_blocks_per_sequence), dtype=torch.int32)

            self.input_tensors.append((k, v, k_cache, v_cache, kv_lengths, block_tables))

    def to_cuda(self, input_tensor):
        return tuple(tensor.cuda() for tensor in input_tensor)

    def call_op(self, input_tensor):
        k, v, k_cache, v_cache, kv_lengths, block_tables = input_tensor
        copy_kv_to_blocked_cache(k, v, k_cache, v_cache, kv_lengths, block_tables)

    def get_gbps(self, input_tensor, runtime):
        k, v, k_cache, v_cache, kv_lengths, block_tables = input_tensor
        total_bytes = (k.numel() + v.numel() + k_cache.numel() + v_cache.numel()) * k.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        k, v, k_cache, v_cache, kv_lengths, block_tables = input_tensor
        # Assuming each element operation is a FLOP
        FLOPS = (k.numel() + v.numel()) * 2  # Load and store operations
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
