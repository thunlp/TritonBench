import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.destindex_copy_kv1 import destindex_copy_kv
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('destindex_copy_kv1', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 18):  # Example sizes, adjust as needed
            seq_len = 2 ** i
            head_num = 64  # Example head number
            head_dim = 64  # Example head dimension
            K = torch.rand(seq_len, head_num, head_dim, dtype=torch.float16)
            DestLoc = torch.randint(0, seq_len, (seq_len,), dtype=torch.int32)
            Out = torch.zeros_like(K)
            self.input_tensors.append((K, DestLoc, Out))

    def to_cuda(self, input_tensor):
        K, DestLoc, Out = input_tensor
        return (K.cuda(), DestLoc.cuda(), Out.cuda())

    def call_op(self, input_tensor):
        K, DestLoc, Out = input_tensor
        destindex_copy_kv(K, DestLoc, Out)
        return Out

    def get_gbps(self, input_tensor, runtime):
        K, DestLoc, Out = input_tensor
        total_bytes = (K.numel() + DestLoc.numel() + Out.numel()) * K.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        K, DestLoc, Out = input_tensor
        FLOPS = 2 * K.numel()  # Assuming each element involves a load and a store
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
