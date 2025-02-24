import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.chunk_delta_fwd import chunk_fwd_h_fn  # Correctly import the function
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('chunk_delta_fwd', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 11):  # Adjust the range as needed for your testing
            B = 2 ** i
            H = 8  # Number of heads, can be adjusted
            T = 128  # Sequence length, can be adjusted
            K = 64  # Key dimension, can be adjusted
            V = 64  # Value dimension, can be adjusted
            BT = 32  # Block size for T, can be adjusted
            k = torch.rand(B, H, T, K, dtype=torch.float16)
            u = torch.rand(B, H, T, V, dtype=torch.float16)
            w = torch.rand(B, H, T, K, dtype=torch.float16)
            initial_state = torch.rand(B, H, K, V, dtype=torch.float16) if i % 2 == 0 else None
            final_state = torch.rand(B, H, K, V, dtype=torch.float16) if i % 2 == 0 else None
            self.input_tensors.append((k, u, w, BT, initial_state, final_state))

    def to_cuda(self, input_tensor):
        k, u, w, BT, initial_state, final_state = input_tensor
        return (k.cuda(), u.cuda(), w.cuda(), BT, 
                initial_state.cuda() if initial_state is not None else None, 
                final_state.cuda() if final_state is not None else None)

    def call_op(self, input_tensor):
        k, u, w, BT, initial_state, final_state = input_tensor
        return chunk_fwd_h_fn(k, w, u, BT, initial_state, final_state)

    def get_gbps(self, input_tensor, runtime):
        k, u, w, BT, initial_state, final_state = input_tensor
        total_bytes = (k.numel() + u.numel() + w.numel()) * k.element_size()
        if initial_state is not None:
            total_bytes += initial_state.numel() * initial_state.element_size()
        if final_state is not None:
            total_bytes += final_state.numel() * final_state.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        k, u, w, BT, initial_state, final_state = input_tensor
        B, H, T, K, V = k.shape[0], k.shape[1], k.shape[2], k.shape[3], u.shape[3]
        FLOPS = 2 * B * H * T * K * V  # Assuming each operation involves 2 FLOPS
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
