import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.llama_ff_triton import kernel_ff
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('llama_ff_triton', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 16):  # Adjust the range as needed for your testing
            batch_size = 2 ** i
            seq_len = 32  # Example sequence length
            dim = 64  # Example dimension
            x = torch.rand((batch_size, seq_len, dim), dtype=torch.float16)
            w1 = torch.rand((dim, dim), dtype=torch.float16)
            w3 = torch.rand((dim, dim), dtype=torch.float16)
            rms_w = torch.rand((dim,), dtype=torch.float16)
            self.input_tensors.append((x, w1, w3, rms_w))

    def to_cuda(self, input_tensor):
        return tuple(tensor.cuda() for tensor in input_tensor)

    def call_op(self, input_tensor):
        x, w1, w3, rms_w = input_tensor
        return kernel_ff(x, w1, w3, rms_w)

    def get_gbps(self, input_tensor, runtime):
        x, w1, w3, rms_w = input_tensor
        total_bytes = (x.numel() + w1.numel() + w3.numel() + rms_w.numel()) * x.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        x, w1, w3, rms_w = input_tensor
        M, K = x.shape[0] * x.shape[1], x.shape[2]
        N = w1.shape[1]
        # Assuming each element in the output is a result of a fused operation
        FLOPS = 2 * M * N * K  # Adjust based on actual operations
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
