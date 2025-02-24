import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.kcache_copy_triton import copy_k_to_blocked_cache
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('kcache_copy_triton', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        for i in range(2, 11):  # Adjust the range for desired sizes
            print(i)
            bsz = 2 ** i
            num_kv_heads = 32
            head_dim = 128
            block_size = 32
            num_blocks = 16
            x = 2
            k = torch.rand((bsz, 1, num_kv_heads, head_dim), dtype=torch.float16)
            k_cache = torch.rand((num_blocks, num_kv_heads, head_dim // x, block_size, x), dtype=torch.float16)
            kv_lengths = torch.randint(0, block_size, (bsz,), dtype=torch.int32)
            block_tables = torch.randint(0, block_size, (bsz, block_size), dtype=torch.int32)
            self.input_tensors.append((k, k_cache, kv_lengths, block_tables))

    def to_cuda(self, input_tensor):
        k, k_cache, kv_lengths, block_tables = input_tensor
        return (k.cuda(), k_cache.cuda(), kv_lengths.cuda(), block_tables.cuda())

    def call_op(self, input_tensor):
        k, k_cache, kv_lengths, block_tables = input_tensor
        copy_k_to_blocked_cache(k, k_cache, kv_lengths, block_tables)

    def get_gbps(self, input_tensor, runtime):
        k, k_cache, _, _ = input_tensor
        total_bytes = k.numel() * k.element_size() + k_cache.numel() * k_cache.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        k, _, _, _ = input_tensor
        FLOPS = k.numel()  # Assuming one operation per element
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
