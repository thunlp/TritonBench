import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.triton_attention import attention
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('triton_attention', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 12):  # Adjust the range as needed for your tests
            print(i)
            size = 2 ** i
            q = torch.rand((64, 8, size, 128), dtype=torch.float16)
            k = torch.rand((64, 8, size, 128), dtype=torch.float16)
            v = torch.rand((64, 8, size, 128), dtype=torch.float16)
            sm_scale = 1.0 / (128 ** 0.5)
            self.input_tensors.append((q, k, v, sm_scale))

    def to_cuda(self, input_tensor):
        q, k, v, sm_scale = input_tensor
        return (q.cuda(), k.cuda(), v.cuda(), sm_scale)

    def call_op(self, input_tensor):
        q, k, v, sm_scale = input_tensor
        return attention(q, k, v, sm_scale)

    def get_gbps(self, input_tensor, runtime):
        q, k, v, _ = input_tensor
        total_bytes = (q.numel() + k.numel() + v.numel() + q.numel()) * q.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        q, _, _, _ = input_tensor
        num_operations = 2 * q.size(2) * q.size(3) * q.size(3)  # Simplified FLOP count for attention
        TFLOPS = num_operations / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
