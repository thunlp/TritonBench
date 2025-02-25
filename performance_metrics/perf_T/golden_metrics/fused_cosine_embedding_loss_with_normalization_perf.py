import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.fused_cosine_embedding_loss_with_normalization import fused_cosine_embedding_loss_with_normalization
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_cosine_embedding_loss_with_normalization', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 21):
            N = 2 ** i
            D = 256
            input1 = torch.randn(N, D, dtype=torch.float32)
            input2 = torch.randn(N, D, dtype=torch.float32)
            target = torch.randint(0, 2, (N,), dtype=torch.float32) * 2 - 1
            self.input_tensors.append((input1, input2, target))

    def to_cuda(self, input_tuple):
        input1, input2, target = input_tuple
        return (input1.cuda(), input2.cuda(), target.cuda())
    
    def call_op(self, input_tuple):
        input1, input2, target = input_tuple
        return fused_cosine_embedding_loss_with_normalization(input1, input2, target)

    def get_gbps(self, input_tensor, runtime):
        input1, input2, target = input_tensor
        element_size = input1.element_size()
        
        input_bytes = (input1.numel() * 10 + target.numel()) * element_size
        output_bytes = 1 * element_size
        total_bytes = (input_bytes + output_bytes)
        
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        input1, input2, target = input_tensor
        N, D = input1.shape
        
        total_flops = 8 * N * D
        
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
