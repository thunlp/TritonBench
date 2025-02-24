import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.mixed_sparse_attention import _triton_mixed_sparse_attention
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('mixed_sparse_attention', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 13):  # Adjust the range for different sizes
            batch_size = 2 ** i
            n_heads = 8
            n_ctx = 128
            d_head = 64
            block_size_M = 64
            q = torch.rand((batch_size, n_heads, n_ctx, d_head), dtype=torch.float16)
            k = torch.rand((batch_size, n_heads, n_ctx, d_head), dtype=torch.float16)
            v = torch.rand((batch_size, n_heads, n_ctx, d_head), dtype=torch.float16)
            seqlens = torch.randint(1, n_ctx, (batch_size,), dtype=torch.int32)
            block_count = torch.randint(1, 10, (batch_size, n_heads, n_ctx // block_size_M), dtype=torch.int32)
            block_offset = torch.randint(0, n_ctx, (batch_size, n_heads, n_ctx // block_size_M, 8), dtype=torch.int32)
            column_count = torch.randint(1, 10, (batch_size, n_heads, n_ctx // block_size_M), dtype=torch.int32)
            column_index = torch.randint(0, n_ctx, (batch_size, n_heads, n_ctx // block_size_M, 16), dtype=torch.int32)
            sm_scale = 1.0 / (d_head ** 0.5)
            self.input_tensors.append((q, k, v, seqlens, block_count, block_offset, column_count, column_index, sm_scale))

    def to_cuda(self, input_tensor):
        return tuple(t.cuda() if isinstance(t, torch.Tensor) else t for t in input_tensor)
    

    def call_op(self, input_tensor):
        q, k, v, seqlens, block_count, block_offset, column_count, column_index, sm_scale = input_tensor
        return _triton_mixed_sparse_attention(q, k, v, seqlens, block_count, block_offset, column_count, column_index, sm_scale)

    def get_gbps(self, input_tensor, runtime):
        q, k, v, *_ = input_tensor
        total_bytes = (q.numel() + k.numel() + v.numel()) * q.element_size() * 2  # Read and write
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        q, *_ = input_tensor
        FLOPS = 2 * q.numel() * q.shape[-1]  # Assuming 2 operations per element
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
