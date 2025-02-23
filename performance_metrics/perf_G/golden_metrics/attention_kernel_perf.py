import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.attention_kernel import _attention_rel_h_rel_w_kernel_aligned_device
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('attention_kernel', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 18):  # Example sizes, adjust as needed
            size = 2 ** i
            Q = torch.rand((1, 1, size, 64), dtype=torch.float16)
            K = torch.rand((1, 1, size, 64), dtype=torch.float16)
            V = torch.rand((1, 1, size, 64), dtype=torch.float16)
            rel_h_w = torch.rand((1, 1, size, 128), dtype=torch.float16)
            Out = torch.empty_like(Q)
            sm_scale = 1.0 / (64 ** 0.5)  # Example scaling factor
            self.input_tensors.append((Q, K, V, rel_h_w, sm_scale, Out))

    def to_cuda(self, input_tensor):
        Q, K, V, rel_h_w, sm_scale, Out = input_tensor
        return (Q.cuda(), K.cuda(), V.cuda(), rel_h_w.cuda(), sm_scale, Out.cuda())

    def call_op(self, input_tensor):
        Q, K, V, rel_h_w, sm_scale, Out = input_tensor
        _attention_rel_h_rel_w_kernel_aligned_device(
            Q, K, V, rel_h_w, sm_scale, Out,
            BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=2
        )
        return Out

    def get_gbps(self, input_tensor, runtime):
        Q, _, _, _, _, _ = input_tensor
        total_bytes = 3 * Q.numel() * Q.element_size()  # Q, K, V are similar in size
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        Q, _, _, _, _, _ = input_tensor
        FLOPS = 2 * Q.size(2) * Q.size(3) * Q.size(2)  # Simplified FLOP count
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
