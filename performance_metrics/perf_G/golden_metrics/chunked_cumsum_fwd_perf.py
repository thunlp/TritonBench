import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.chunked_cumsum_fwd import _chunk_cumsum_fwd
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('chunked_cumsum_fwd', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(3, 11):  # Adjust the range for different sizes
            batch = 128 * i
            seqlen = 128 * i
            nheads = 128 * (i // 2)
            chunk_size = 128 * (i // 3)
            dt = torch.rand(batch, seqlen, nheads, dtype=torch.float32)
            A = torch.rand(nheads, dtype=torch.float32)
            dt_bias = torch.rand(nheads, dtype=torch.float32)
            self.input_tensors.append((dt, A, chunk_size, dt_bias))

    def to_cuda(self, input_tensor):
        dt, A, chunk_size, dt_bias = input_tensor
        return (dt.cuda(), A.cuda(), chunk_size, dt_bias.cuda())

    def call_op(self, input_tensor):
        dt, A, chunk_size, dt_bias = input_tensor
        return _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias)

    def get_gbps(self, input_tensor, runtime):
        dt, A, chunk_size, dt_bias = input_tensor
        total_bytes = dt.numel() * dt.element_size() + A.numel() * A.element_size() + dt_bias.numel() * dt_bias.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        dt, A, chunk_size, dt_bias = input_tensor
        batch, seqlen, nheads = dt.shape
        FLOPS = batch * seqlen * nheads * 2  # Assuming 2 operations per element
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
