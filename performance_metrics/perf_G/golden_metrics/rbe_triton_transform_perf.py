import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.rbe_triton_transform import rbe_triton_wrapper
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('rbe_triton_transform', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 16):  # Adjust the range as needed for testing
            M = 2 ** i
            K = 1024
            batch_size = 2  # Example batch size
            input_tensor = torch.rand((batch_size, M, K), dtype=torch.float16)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()

    def call_op(self, input_tensor):
        pos = 0  # Example starting position
        return rbe_triton_wrapper(input_tensor, pos)

    def get_gbps(self, input_tensor, runtime):
        x = input_tensor
        total_bytes = 2 * x.numel() * x.element_size()  # Read and write
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        x = input_tensor
        FLOPS = 4 * x.numel()  # 4 operations per element (2 multiplications and 2 additions)
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
