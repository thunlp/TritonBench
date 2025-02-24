import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.chunk_gate_recurrence import chunk_gate_recurrent
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('chunk_gate_recurrence', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 11):  # Adjust the range as needed for your testing
            B = 4
            H = 4
            N = 2 ** i
            D_k = 128
            D_v = 64
            kv = torch.rand(B, H, N, D_k, D_v, dtype=torch.float16)
            cross_decay = torch.rand(B, H, N, dtype=torch.float16)
            self.input_tensors.append((kv, cross_decay))

    def to_cuda(self, input_tensor):
        kv, cross_decay = input_tensor
        return kv.cuda(), cross_decay.cuda()

    def call_op(self, input_tensor):
        kv, cross_decay = input_tensor
        return chunk_gate_recurrent(kv, cross_decay)

    def get_gbps(self, input_tensor, runtime):
        kv, _ = input_tensor
        total_bytes = kv.numel() * kv.element_size() * 2  # Read and write
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        kv, _ = input_tensor
        # Assuming each element involves a multiply and an add
        FLOPS = 2 * kv.numel()
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
