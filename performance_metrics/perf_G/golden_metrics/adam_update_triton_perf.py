import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.adam_update_triton import update_fn
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('adam_update_triton', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):
            size = 2 ** i
            p = torch.rand(size, dtype=torch.float32)
            grad = torch.rand(size, dtype=torch.float32)
            exp_avg = torch.rand(size, dtype=torch.float32)
            self.input_tensors.append((p, grad, exp_avg))

    def to_cuda(self, input_tensor):
        p, grad, exp_avg = input_tensor
        return (p.cuda(), grad.cuda(), exp_avg.cuda())

    def call_op(self, input_tensor):
        p, grad, exp_avg = input_tensor
        lr = 0.001
        wd = 0.01
        beta1 = 0.9
        beta2 = 0.999
        update_fn(p, grad, exp_avg, lr, wd, beta1, beta2)

    def get_gbps(self, input_tensor, runtime):
        p, grad, exp_avg = input_tensor
        total_bytes = 3 * p.numel() * p.element_size()  # p, grad, exp_avg
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        p, grad, exp_avg = input_tensor
        FLOPS = 2 * p.numel()  # Simplified estimation
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config()
    op_perf.run_benchmark()
