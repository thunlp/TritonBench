import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.cache_transform_triton import get_xine_cache
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('cache_transform_triton', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 20):  # Adjust the range as needed for testing
            num_seqs = 2 ** i
            hidden_dim = 256  # Example hidden dimension, adjust as needed
            lengths = torch.randint(1, 10, (num_seqs,), dtype=torch.int32)
            cos_cache = torch.rand((num_seqs, hidden_dim), dtype=torch.float16)
            sin_cache = torch.rand((num_seqs, hidden_dim), dtype=torch.float16)
            self.input_tensors.append((lengths, cos_cache, sin_cache))

    def to_cuda(self, input_tensor):
        lengths, cos_cache, sin_cache = input_tensor
        return (lengths.cuda(), cos_cache.cuda(), sin_cache.cuda())

    def call_op(self, input_tensor):
        lengths, cos_cache, sin_cache = input_tensor
        return get_xine_cache(lengths, cos_cache, sin_cache, is_prompts=False)

    def get_gbps(self, input_tensor, runtime):
        lengths, cos_cache, sin_cache = input_tensor
        total_bytes = (cos_cache.numel() + sin_cache.numel()) * cos_cache.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        lengths, cos_cache, sin_cache = input_tensor
        # Assuming each element involves a few operations, adjust as needed
        FLOPS = 2 * (cos_cache.numel() + sin_cache.numel())
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
