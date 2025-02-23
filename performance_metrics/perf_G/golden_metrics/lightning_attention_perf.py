import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.lightning_attention import lightning_attn2_no_decay
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('lightning_attention', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 20):  # Adjust the range for different sizes
            b = 2  # batch size
            h = 8  # number of heads
            n = 2 ** i  # sequence length
            d = 64  # dimension per head
            e = 64  # output dimension
            q = torch.rand((b, h, n, d), dtype=torch.float16)
            k = torch.rand((b, h, n, d), dtype=torch.float16)
            v = torch.rand((b, h, n, e), dtype=torch.float16)
            self.input_tensors.append((q, k, v))

    def to_cuda(self, input_tensor):
        q, k, v = input_tensor
        return (q.cuda(), k.cuda(), v.cuda())

    def call_op(self, input_tensor):
        q, k, v = input_tensor
        return lightning_attn2_no_decay(q, k, v)

    def get_gbps(self, input_tensor, runtime):
        q, k, v = input_tensor
        total_bytes = (q.numel() + k.numel() + v.numel() + q.numel() * v.size(-1) // q.size(-1)) * q.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        q, k, v = input_tensor
        b, h, n, d = q.shape
        e = v.shape[-1]
        # FLOPS calculation: 2 * b * h * n * d * e (for qk and kv operations)
        FLOPS = 2 * b * h * n * d * e
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
