import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Correctly import the operator
from TritonBench_v1.attn_fwd_triton import forward
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('attn_fwd_triton', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 12):  # Example sizes, adjust as needed
            batch_size = 2 ** i
            num_heads = 8
            seq_len = 128
            head_dim = 64
            q = torch.rand((batch_size, num_heads, seq_len, head_dim), dtype=torch.float16)
            k = torch.rand((batch_size, num_heads, seq_len, head_dim), dtype=torch.float16)
            v = torch.rand((batch_size, num_heads, seq_len, head_dim), dtype=torch.float16)
            q_scale = torch.tensor(1.0, dtype=torch.float32)
            k_scale = torch.tensor(1.0, dtype=torch.float32)
            self.input_tensors.append((q, k, v, q_scale, k_scale))

    def to_cuda(self, input_tensor):
        q, k, v, q_scale, k_scale = input_tensor
        return (q.cuda(), k.cuda(), v.cuda(), q_scale.cuda(), k_scale.cuda())

    def call_op(self, input_tensor):
        q, k, v, q_scale, k_scale = input_tensor
        return forward(q, k, v, q_scale, k_scale)

    def get_gbps(self, input_tensor, runtime):
        q, k, v, q_scale, k_scale = input_tensor
        total_bytes = (q.numel() + k.numel() + v.numel() + q.numel()) * q.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        q, k, v, q_scale, k_scale = input_tensor
        batch_size, num_heads, seq_len, head_dim = q.shape
        FLOPS = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
