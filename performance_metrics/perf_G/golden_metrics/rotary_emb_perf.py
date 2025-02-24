import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.rotary_emb import rotary_emb_fwd
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('rotary_emb', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 18):  # Adjust the range as needed for your testing
            seq_len = 2 ** i
            head_num = 8  # Example head number
            head_dim = 64  # Example head dimension
            q = torch.rand((seq_len, head_num, head_dim), dtype=torch.float16)
            k = torch.rand((seq_len, head_num, head_dim), dtype=torch.float16)
            cos = torch.rand((seq_len, head_dim), dtype=torch.float16)
            sin = torch.rand((seq_len, head_dim), dtype=torch.float16)
            self.input_tensors.append((q, k, cos, sin))

    def to_cuda(self, input_tensor):
        q, k, cos, sin = input_tensor
        return q.cuda(), k.cuda(), cos.cuda(), sin.cuda()

    def call_op(self, input_tensor):
        q, k, cos, sin = input_tensor
        rotary_emb_fwd(q, k, cos, sin)

    def get_gbps(self, input_tensor, runtime):
        q, k, cos, sin = input_tensor
        total_bytes = (q.numel() + k.numel() + cos.numel() + sin.numel()) * q.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        q, k, cos, sin = input_tensor
        # Assuming each element in q and k undergoes a few operations (e.g., 4 operations per element)
        FLOPS = 4 * (q.numel() + k.numel())
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
