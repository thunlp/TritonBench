import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Correctly import the operator
from TritonBench_v1.diag_ssm_triton import diag_ssm_forward_triton
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('diag_ssm_triton', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 20):  # Adjust the range as needed for testing
            length = 2 ** i
            batch_size = 2 ** 5  # Example batch size
            dim = 2 ** 5  # Example dimension size
            s = torch.rand(batch_size, dim, dtype=torch.float16)
            x = torch.rand(length, batch_size, dim, dtype=torch.float16)
            Lambda = torch.rand(dim, dtype=torch.float16)
            self.input_tensors.append((s, x, Lambda))

    def to_cuda(self, input_tensor):
        s, x, Lambda = input_tensor
        return (s.cuda(), x.cuda(), Lambda.cuda())

    def call_op(self, input_tensor):
        s, x, Lambda = input_tensor
        return diag_ssm_forward_triton(s, x, Lambda)

    def get_gbps(self, input_tensor, runtime):
        s, x, Lambda = input_tensor
        total_bytes = (s.numel() + x.numel() + Lambda.numel()) * s.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        s, x, Lambda = input_tensor
        FLOPS = 2 * x.numel()  # Assuming 2 operations per element in x
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
