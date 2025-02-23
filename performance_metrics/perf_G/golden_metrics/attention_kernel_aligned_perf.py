import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.attention_kernel_aligned import _attention_rel_h_rel_w_kernel_aligned_device
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('attention_kernel_aligned', dtype=dtype, is_backward=is_backward, **kwargs)
        self.input_tensors = []

    def get_input_tensors(self):
        for i in range(2, 20):  # Example sizes, you may adjust based on your needs
            size = 2 ** i
            q = torch.rand((1, 1, size, 64), dtype=torch.float16)  # Example dimensions
            k = torch.rand((1, 1, size, 64), dtype=torch.float16)
            v = torch.rand((1, 1, size, 64), dtype=torch.float16)
            rel_h_w = torch.rand((1, 1, size, 128), dtype=torch.float16)
            o = torch.empty_like(q)
            sm_scale = 1.0 / (64 ** 0.5)
            self.input_tensors.append((q, k, v, rel_h_w, sm_scale, o))

    def to_cuda(self, input_tensor):
        cuda_tensors = [t.cuda() for t in input_tensor[:4]]
        cuda_tensors.append(input_tensor[4])
        cuda_tensors.append(input_tensor[5].cuda())
        cuda_tensors = tuple(cuda_tensors)
        return cuda_tensors

    def call_op(self, input_tensor):
        q, k, v, rel_h_w, sm_scale, o = input_tensor
        _attention_rel_h_rel_w_kernel_aligned_device(q, k, v, rel_h_w, sm_scale, o, BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=2)
        return o

    def get_gbps(self, input_tensor, runtime):
        q, _, _, _, _, _ = input_tensor
        total_bytes = 4 * q.numel() * q.element_size()  # q, k, v, and o
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        q, _, _, _, _, _ = input_tensor
        FLOPS = 2 * q.size(2) * q.size(3) * q.size(2)  # Simplified FLOPS calculation
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
