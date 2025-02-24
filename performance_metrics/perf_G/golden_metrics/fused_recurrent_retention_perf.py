import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.fused_recurrent_retention import fused_recurrent_retention
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_recurrent_retention', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 18):  # Adjust the range for different sizes
            size = 2 ** i
            batch_size = 2
            n_heads = 2
            seq_len = size
            d_head_qk = 64
            d_head_v = 64
            q = torch.rand(batch_size, n_heads, seq_len, d_head_qk, dtype=torch.float32)
            k = torch.rand(batch_size, n_heads, seq_len, d_head_qk, dtype=torch.float32)
            v = torch.rand(batch_size, n_heads, seq_len, d_head_v, dtype=torch.float32)
            initial_state = torch.rand(batch_size, n_heads, d_head_qk, d_head_v, dtype=torch.float32)
            self.input_tensors.append((q, k, v, initial_state))

    def to_cuda(self, input_tensor):
        q, k, v, initial_state = input_tensor
        return (q.cuda(), k.cuda(), v.cuda(), initial_state.cuda())

    def call_op(self, input_tensor):
        q, k, v, initial_state = input_tensor
        return fused_recurrent_retention(q, k, v, initial_state, output_final_state=True)

    def get_gbps(self, input_tensor, runtime):
        q, k, v, initial_state = input_tensor
        total_bytes = (q.numel() + k.numel() + v.numel() + initial_state.numel()) * q.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        q, k, v, initial_state = input_tensor
        # Assuming each element in q, k, v is involved in a multiply-add operation
        FLOPS = 2 * (q.numel() + k.numel() + v.numel())
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
