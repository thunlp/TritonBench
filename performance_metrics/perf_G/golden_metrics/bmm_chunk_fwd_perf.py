import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.bmm_chunk_fwd import _bmm_chunk_fwd
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('bmm_chunk_fwd', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 16):  # Adjust the range for desired sizes
            batch_size = 2 ** i
            seqlen = 128
            k = 64  # Fixed K dimension for simplicity
            chunk_size = 64  # Example chunk size
            a = torch.rand((batch_size, seqlen, k), dtype=torch.float16)
            b = torch.rand((batch_size, seqlen, k), dtype=torch.float16)
            self.input_tensors.append((a, b, chunk_size))

    def to_cuda(self, input_tensor):
        a, b, chunk_size = input_tensor
        return (a.cuda(), b.cuda(), chunk_size)

    def call_op(self, input_tensor):
        a, b, chunk_size = input_tensor
        return _bmm_chunk_fwd(a, b, chunk_size)

    def get_gbps(self, input_tensor, runtime):
        a, b, chunk_size = input_tensor
        total_bytes = 2 * a.numel() * a.element_size()  # Read a and b
        total_bytes += a.shape[0] * (a.shape[1] // chunk_size) * chunk_size * chunk_size * a.element_size()  # Write output
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        a, b, chunk_size = input_tensor
        batch, seqlen, k = a.shape
        FLOPS = 2 * batch * seqlen * k * (seqlen // chunk_size)  # 2 * M * N * K for each chunk
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
