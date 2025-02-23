import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.rotary_transform_ops import apply_rotary
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('rotary_transform_ops', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 16):  # Example range, adjust based on your needs
            batch_size = 2 ** i
            seqlen = 128  # Example sequence length
            nheads = 8  # Example number of heads
            headdim = 64  # Example head dimension
            rotary_dim = headdim // 2
            x = torch.rand((batch_size, seqlen, nheads, headdim), dtype=torch.float32)
            cos = torch.rand((seqlen, rotary_dim), dtype=torch.float32)
            sin = torch.rand((seqlen, rotary_dim), dtype=torch.float32)
            self.input_tensors.append((x, cos, sin))

    def to_cuda(self, input_tensor):
        x, cos, sin = input_tensor
        return (x.cuda(), cos.cuda(), sin.cuda())

    def call_op(self, input_tensor):
        x, cos, sin = input_tensor
        return apply_rotary(x, cos, sin)

    def get_gbps(self, input_tensor, runtime):
        x, _, _ = input_tensor
        total_bytes = 2 * x.numel() * x.element_size()  # Read and write
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        x, _, _ = input_tensor
        FLOPS = 4 * x.numel()  # Assuming 4 FLOPs per element (2 multiplications and 2 additions)
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
