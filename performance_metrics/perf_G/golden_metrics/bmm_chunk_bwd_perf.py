import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.bmm_chunk_bwd import _bmm_chunk_bwd
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('bmm_chunk_bwd', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 16):  # Adjust the range as needed for your testing
            batch_size = 2 ** i
            seqlen = 128  # Example sequence length
            k = 64  # Example dimension size
            chunk_size = 32  # Example chunk size
            a = torch.rand((batch_size, seqlen, k), dtype=torch.float16)
            dout = torch.rand((batch_size, seqlen // chunk_size, chunk_size, k), dtype=torch.float16)
            self.input_tensors.append((a, dout))

    def to_cuda(self, input_tensor):
        a, dout = input_tensor
        return a.cuda(), dout.cuda()

    def call_op(self, input_tensor):
        a, dout = input_tensor
        return _bmm_chunk_bwd(a, dout)

    def get_gbps(self, input_tensor, runtime):
        a, dout = input_tensor
        total_bytes = (a.numel() + dout.numel()) * a.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        a, dout = input_tensor
        batch_size, seqlen, k = a.shape
        nchunks = dout.shape[1]
        FLOPS = 2 * batch_size * seqlen * k * nchunks  # Example calculation
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
