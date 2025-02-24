import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.f8_conversion_utils import f8_to_f16, f16_to_f8
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('f8_conversion_utils', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        for i in range(2, 28):
            size = 2 ** i
            # Create input tensors for both f8_to_f16 and f16_to_f8
            input_tensor_f8 = torch.randint(-128, 127, (size,), dtype=torch.int8)
            input_tensor_f16 = torch.rand(size, dtype=torch.float16)
            self.input_tensors.append((input_tensor_f8, input_tensor_f16))

    def to_cuda(self, input_tensor):
        return (input_tensor[0].cuda(), input_tensor[1].cuda())

    def call_op(self, input_tensor):
        return f8_to_f16(input_tensor[0])

    def get_gbps(self, input_tensor, runtime):
        x = input_tensor[0]
        total_bytes = x.numel() * x.element_size() * 2  # Read and write
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        # Assuming the conversion itself doesn't involve FLOPs, just data movement
        return 0.0


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
