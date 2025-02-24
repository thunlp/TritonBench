import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.int8_quantization import per_block_int8
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('int8_quantization', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 10):  # Adjust the range as needed for your testing
            size = 2 ** i
            q = torch.rand((size, size, size), dtype=torch.float32)
            k = torch.rand((size, size, size), dtype=torch.float32)
            self.input_tensors.append((q, k))

    def to_cuda(self, input_tensor):
        q, k = input_tensor
        return q.cuda(), k.cuda()

    def call_op(self, input_tensor):
        q, k = input_tensor
        return per_block_int8(q, k)

    def get_gbps(self, input_tensor, runtime):
        q, _ = input_tensor
        total_bytes = 2 * q.numel() * q.element_size()  # q and k are both processed
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        q, _ = input_tensor
        FLOPS = 2 * q.numel()  # Assuming each element is processed twice
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
