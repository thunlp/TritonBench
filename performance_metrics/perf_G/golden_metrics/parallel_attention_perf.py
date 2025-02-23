import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.parallel_attention import parallel_rebased
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('parallel_attention', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 12):  # Adjust the range as needed for your testing
            B = 64  # Batch size
            H = 8  # Number of heads
            L = 2 ** i  # Sequence length
            D_head_K = 128  # Dimension of head for K
            D_head_V = 128  # Dimension of head for V
            q = torch.rand(B, H, L, D_head_K, dtype=torch.float16)
            k = torch.rand(B, H, L, D_head_K, dtype=torch.float16)
            v = torch.rand(B, H, L, D_head_V, dtype=torch.float16)
            self.input_tensors.append((q, k, v))

    def to_cuda(self, input_tensor):
        q, k, v = input_tensor
        return q.cuda(), k.cuda(), v.cuda()

    def call_op(self, input_tensor):
        q, k, v = input_tensor
        return parallel_rebased(q, k, v)

    def get_gbps(self, input_tensor, runtime):
        q, k, v = input_tensor
        total_bytes = (q.numel() + k.numel() + v.numel()) * q.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        q, k, v = input_tensor
        B, H, L, D_head_K = q.shape
        D_head_V = v.shape[-1]
        # Assuming FLOPS is calculated as 2 * B * H * L * D_head_K * D_head_V
        FLOPS = 2 * B * H * L * D_head_K * D_head_V
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
