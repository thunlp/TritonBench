import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.attention_fwd_triton3 import _forward
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('attention_fwd_triton3', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 12):  # Example sizes, adjust as needed
            size = 2 ** i
            q = torch.rand(32, 8, size, 128, dtype=torch.float32)  # Example dimensions
            k = torch.rand(32, 8, size, 128, dtype=torch.float32)
            v = torch.rand(32, 8, size, 128, dtype=torch.float32)
            sm_scale = 1.0
            self.input_tensors.append((q, k, v, sm_scale))

    def to_cuda(self, input_tensor):
        q, k, v, sm_scale = input_tensor
        return (q.cuda(), k.cuda(), v.cuda(), sm_scale)

    def call_op(self, input_tensor):
        q, k, v, sm_scale = input_tensor
        o = torch.empty_like(q)
        m = torch.empty(q.shape[0], q.shape[1], q.shape[2], dtype=torch.float32, device=q.device)
        l = torch.empty(q.shape[0], q.shape[1], q.shape[2], dtype=torch.float32, device=q.device)
        return _forward(q, k, v, sm_scale, o=o, m=m, l=l, end=True)

    def get_gbps(self, input_tensor, runtime):
        q, k, v, sm_scale = input_tensor
        total_bytes = (q.numel() + k.numel() + v.numel() + q.numel()) * q.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        q, k, v, sm_scale = input_tensor
        # Assuming each element in q, k, v contributes to a FLOP
        FLOPS = 2 * q.numel() * k.shape[-1]  # Example calculation
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
