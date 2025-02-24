import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.index_select_bwd import index_select_cat_bwd
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('index_select_bwd', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        for i in range(10, 22):  # Adjust the range as needed for testing
            num_rows = 2 ** i
            num_indices = 2 ** (i - 1)  # Ensure num_indices <= num_rows
            num_cols = 256  # Fixed number of columns for simplicity
            grad_source = torch.rand((num_rows, num_cols), dtype=torch.float16)
            index = torch.randint(0, num_rows, (num_indices,), dtype=torch.int32)
            grad_output = torch.rand((num_indices, num_cols), dtype=torch.float16)
            self.input_tensors.append((grad_source, index, grad_output))

    def to_cuda(self, input_tensor):
        grad_source, index, grad_output = input_tensor
        return (grad_source.cuda(), index.cuda(), grad_output.cuda())

    def call_op(self, input_tensor):
        grad_source, index, grad_output = input_tensor
        index_select_cat_bwd(grad_source, index, grad_output)

    def get_gbps(self, input_tensor, runtime):
        grad_source, index, grad_output = input_tensor
        total_bytes = grad_source.numel() * grad_source.element_size() / 2 + grad_output.numel() * grad_output.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        grad_source, index, grad_output = input_tensor
        # Assuming each element in grad_output is added to grad_source
        FLOPS = 2 * grad_output.numel()
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
