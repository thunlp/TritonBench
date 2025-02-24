import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.rope_transform import rope_forward
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('rope_transform', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 14):  # Adjust the range as needed for different sizes
            batch_size = 2 ** i
            seq_len = 128  # Example sequence length
            n_q_head = 8  # Example number of query heads
            n_kv_head = 8  # Example number of key/value heads
            head_dim = 64  # Example head dimension

            q = torch.rand(batch_size, seq_len, n_q_head, head_dim, dtype=torch.float16)
            k = torch.rand(batch_size, seq_len, n_kv_head, head_dim, dtype=torch.float16)
            cos = torch.rand(seq_len, head_dim // 2, dtype=torch.float16)
            sin = torch.rand(seq_len, head_dim // 2, dtype=torch.float16)

            self.input_tensors.append((q, k, cos, sin))

    def to_cuda(self, input_tensor):
        q, k, cos, sin = input_tensor
        return q.cuda(), k.cuda(), cos.cuda(), sin.cuda()

    def call_op(self, input_tensor):
        q, k, cos, sin = input_tensor
        return rope_forward(q, k, cos, sin)

    def get_gbps(self, input_tensor, runtime):
        q, k, cos, sin = input_tensor
        total_bytes = (q.numel() + k.numel() + cos.numel() + sin.numel()) * q.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        q, k, cos, sin = input_tensor
        # Assuming each element in q and k undergoes a few operations
        FLOPS = 4 * (q.numel() + k.numel())  # Example: 4 operations per element
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
