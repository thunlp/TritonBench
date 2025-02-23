import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.quantize_kv_copy import destindex_copy_quantize_kv
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('quantize_kv_copy', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 16):  # Adjust the range as needed for testing
            seq_len = 2 ** i
            head_num = 8  # Example head number
            head_dim = 64  # Example head dimension, must be divisible by 8
            K = torch.rand((seq_len, head_num, head_dim), dtype=torch.float16)
            DestLoc = torch.randint(0, seq_len, (seq_len,), dtype=torch.int32)
            Out = torch.empty_like(K, dtype=torch.int8)
            Out_scale = torch.empty((seq_len, head_num, head_dim // 8), dtype=torch.float16)
            self.input_tensors.append((K, DestLoc, Out, Out_scale))

    def to_cuda(self, input_tensor):
        K, DestLoc, Out, Out_scale = input_tensor
        return (K.cuda(), DestLoc.cuda(), Out.cuda(), Out_scale.cuda())

    def call_op(self, input_tensor):
        K, DestLoc, Out, Out_scale = input_tensor
        destindex_copy_quantize_kv(K, DestLoc, Out, Out_scale)
        return Out, Out_scale

    def get_gbps(self, input_tensor, runtime):
        K, _, _, _ = input_tensor
        total_bytes = K.numel() * K.element_size() + K.numel() // 8 * 4  # K and Out_scale
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        K, _, _, _ = input_tensor
        FLOPS = 2 * K.numel()  # Assuming 2 FLOPS per element for quantization
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
