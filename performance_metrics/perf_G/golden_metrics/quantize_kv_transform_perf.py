import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.quantize_kv_transform import destindex_copy_quantize_kv
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('quantize_kv_transform', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 18):  # Example sizes, adjust as needed
            seq_len = 2 ** i
            head_num = 64  # Example head number
            head_dim = 64  # Example head dimension
            K = torch.rand(seq_len, head_num, head_dim, dtype=torch.float16)
            DestLoc = torch.randint(0, seq_len, (seq_len,), dtype=torch.int32)
            Out = torch.empty_like(K, dtype=torch.int8)
            Out_scale = torch.randn((seq_len, head_num, 1), dtype=torch.float16)
            self.input_tensors.append((K, DestLoc, Out, Out_scale))

    def to_cuda(self, input_tensor):
        K, DestLoc, Out, Out_scale = input_tensor
        return (K.cuda(), DestLoc.cuda(), Out.cuda(), Out_scale.cuda())

    def call_op(self, input_tensor):
        K, DestLoc, Out, Out_scale = input_tensor
        destindex_copy_quantize_kv(K, DestLoc, Out, Out_scale)
        return Out, Out_scale

    def get_gbps(self, input_tensor, runtime):
        K, DestLoc, Out, Out_scale = input_tensor
        total_bytes = K.numel() * K.element_size() + DestLoc.numel() * DestLoc.element_size() + Out.numel() * Out.element_size() + Out_scale.numel() * Out_scale.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        K, DestLoc, Out, Out_scale = input_tensor
        # Assuming each element in K is involved in a multiply and an add operation
        FLOPS = 2 * K.numel()
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
