import sys
import os
import math
import torch
import triton
import triton.language as tl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Correctly import the operator
from TritonBench_v1.pow_scalar_tensor import pow_func_scalar_tensor_wrapper_rank_1
from performance_utils import Performance_Metrics, do_bench_config

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('pow_scalar_tensor', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 30):
            size = 2 ** i
            in0 = torch.rand(size, dtype=torch.float16)
            out0 = torch.empty(size, dtype=torch.float16)
            self.input_tensors.append((in0, out0))

    def to_cuda(self, input_tensor):
        # return input_tensor.cuda()
        in0, out0 = input_tensor
        return (in0.cuda(), out0.cuda())

    def call_op(self, input_tensor):
        # # Create an output tensor with the same size as input_tensor
        # output_tensor = torch.empty_like(input_tensor)
        # # Call the operator
        # return pow_func_scalar_tensor_wrapper_rank_1(2.0, input_tensor, out0=output_tensor)
        in0, out0 = input_tensor
        return pow_func_scalar_tensor_wrapper_rank_1(2.0, in0, out0=out0)

    def get_gbps(self, input_tensor, runtime):

        # total_bytes = 2 * input_tensor.numel() * input_tensor.element_size()
        in0, out0 = input_tensor
        total_bytes = 2 * in0.numel() * in0.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        # FLOPS = input_tensor.numel()  # One operation per element
        in0, out0 = input_tensor
        FLOPS = in0.numel()
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
