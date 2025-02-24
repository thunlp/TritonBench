import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.masked_add_cuda import masked_add
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('masked_add_cuda', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):
            size = 2 ** i
            grad = torch.rand(size, dtype=torch.float32)
            p_data = torch.rand(size, dtype=torch.float32)
            p_mask = torch.randint(0, 2, (size,), dtype=torch.bool)
            self.input_tensors.append((grad, p_data, p_mask))

    def to_cuda(self, input_tensor):
        grad, p_data, p_mask = input_tensor
        return grad.cuda(), p_data.cuda(), p_mask.cuda()

    def call_op(self, input_tensor):
        grad, p_data, p_mask = input_tensor
        alpha = 0.1  # Example alpha value
        masked_add(grad, p_data, p_mask, alpha)

    def get_gbps(self, input_tensor, runtime):
        grad, p_data, p_mask = input_tensor
        total_bytes = (grad.numel() + p_data.numel() + p_mask.numel()) * grad.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        grad, p_data, _ = input_tensor
        FLOPS = 2 * grad.numel()  # Each element involves a multiply and an add
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
