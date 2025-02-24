import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.var_len_copy import launch_var_len_copy_triton
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('var_len_copy', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 20):  # Adjust the range as needed for your testing
            size = 2 ** i
            old_a_start = torch.randint(0, size, (size,), dtype=torch.int32)
            old_a_len = torch.randint(1, 256, (size,), dtype=torch.int32)  # Assuming max length is 256
            old_a_location = torch.rand(size * 256, dtype=torch.float32)  # Assuming max length is 256
            new_a_start = torch.randint(0, size, (size,), dtype=torch.int32)
            new_a_location = torch.zeros_like(old_a_location)
            self.input_tensors.append((old_a_start, old_a_len, old_a_location, new_a_start, new_a_location))

    def to_cuda(self, input_tensor):
        return tuple(tensor.cuda() for tensor in input_tensor)

    def call_op(self, input_tensor):
        launch_var_len_copy_triton(*input_tensor)

    def get_gbps(self, input_tensor, runtime):
        old_a_len = input_tensor[1]
        old_a_location = input_tensor[2]
        total_bytes = old_a_len.sum().item() * old_a_location.element_size() * 2
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        # Assuming each element copy is a single operation
        old_a_len = input_tensor[1]
        total_operations = old_a_len.sum().item()
        TFLOPS = total_operations / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
