import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.index_select_cat import index_select_cat_fwd
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('index_select_cat', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(10, 21):  # Adjust the range as needed for your performance testing
            num_rows = 2 ** i
            num_cols = 256  # Fixed column size for simplicity
            source = torch.rand((num_rows, num_cols), dtype=torch.float32)
            index = torch.randint(0, num_rows, (num_rows // 2,), dtype=torch.int64)
            output = torch.empty_like(source)
            self.input_tensors.append((output, source, index))

    def to_cuda(self, input_tensor):
        output, source, index = input_tensor
        return (output.cuda(), source.cuda(), index.cuda())

    def call_op(self, input_tensor):
        output, source, index = input_tensor
        return index_select_cat_fwd(output, source, index)

    def get_gbps(self, input_tensor, runtime):
        output, source, index = input_tensor
        total_bytes = (source.numel()) * source.element_size() / 2 + (output.numel()) * output.element_size() / 2
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        output, source, index = input_tensor
        # Assuming each index selection and copy is a single operation
        FLOPS = index.numel() * source.size(1)
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
