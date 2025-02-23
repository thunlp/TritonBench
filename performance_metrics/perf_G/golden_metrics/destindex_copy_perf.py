import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.destindex_copy import destindex_copy_kv
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('destindex_copy', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 24):  # Adjust the range as needed)
            seq_len = 2 ** i
            head_num = 1  # Example head number
            head_dim = 64  # Example head dimension
            KV_nope = torch.rand(seq_len, head_num, head_dim, dtype=torch.float16)
            KV_rope = torch.rand(seq_len, head_num, head_dim, dtype=torch.float16)
            DestLoc = torch.randint(0, seq_len, (seq_len,), dtype=torch.int32)
            O_nope = torch.zeros_like(KV_nope)
            O_rope = torch.zeros_like(KV_rope)
            self.input_tensors.append((KV_nope, KV_rope, DestLoc, O_nope, O_rope))

    def to_cuda(self, input_tensor):
        return tuple(tensor.cuda() for tensor in input_tensor)

    def call_op(self, input_tensor):
        KV_nope, KV_rope, DestLoc, O_nope, O_rope = input_tensor
        destindex_copy_kv(KV_nope, KV_rope, DestLoc, O_nope, O_rope)
        return O_nope, O_rope

    def get_gbps(self, input_tensor, runtime):
        KV_nope, KV_rope, _, O_nope, O_rope = input_tensor
        total_bytes = 2 * (KV_nope.numel() + KV_rope.numel() + O_nope.numel() + O_rope.numel()) * KV_nope.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        # This operation doesn't perform floating-point operations, so TFLOPS is not applicable
        return 0.0


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
