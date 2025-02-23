import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.block_sparse_attn import block_sparse_attention
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('block_sparse_attn', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 16):  # Example range, adjust as needed
            B = 2  # Batch size
            H = 8  # Number of heads
            M = 2 ** i  # Query sequence length
            D = 64  # Head size
            N = M  # Max sequence length for kv cache
            L = 1  # Number of sparse layouts
            BLOCK_M = 16
            BLOCK_N = 16
            BLOCK_D = 16
            NUM_D_BLOCKS = D // BLOCK_D
            EVEN_M = M % BLOCK_M == 0
            EVEN_N = N % BLOCK_N == 0

            Q = torch.rand((B, H, M, D), dtype=torch.float16)
            K = torch.rand((B, H, N, D), dtype=torch.float16)
            V = torch.rand((B, H, N, D), dtype=torch.float16)
            layout_csr_row_indices = torch.randint(0, N // BLOCK_M + 1, (L, N // BLOCK_M + 1), dtype=torch.int32)
            layout_csr_col_indices = torch.randint(0, N // BLOCK_N, (L, N // BLOCK_M * N // BLOCK_N), dtype=torch.int32)
            layout_csr_row_stride_h = N // BLOCK_M + 1
            layout_csr_col_stride_h = N // BLOCK_M * N // BLOCK_N
            num_layout = L
            softmax_scale = 1.0
            num_heads = H
            num_kv_heads = H
            total_seq_len = M

            input_tensor = (Q, K, V, layout_csr_row_indices, layout_csr_col_indices, layout_csr_row_stride_h, layout_csr_col_stride_h,
                            num_layout, softmax_scale, num_heads, num_kv_heads, total_seq_len, BLOCK_M, EVEN_M, BLOCK_N, EVEN_N, BLOCK_D, NUM_D_BLOCKS)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        return tuple(tensor.cuda() if isinstance(tensor, torch.Tensor) else tensor for tensor in input_tensor)

    def call_op(self, input_tensor):
        return block_sparse_attention(*input_tensor)

    def get_gbps(self, input_tensor, runtime):
        Q, K, V, *_ = input_tensor
        total_bytes = (Q.numel() + K.numel() + V.numel()) * Q.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        Q, *_ = input_tensor
        FLOPS = 2 * Q.size(0) * Q.size(1) * Q.size(2) * Q.size(3)
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
