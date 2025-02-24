import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.multinomial_sampling import multinomial_sampling
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('multinomial_sampling', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 20):  # Adjust the range as needed for your tests
            batch_size = 2 ** i
            num_tokens = 2 ** 10  # Example token size, adjust as needed
            scores = torch.rand((batch_size, num_tokens), dtype=torch.float32)
            seeds = torch.randint(0, 2**31, (batch_size,), dtype=torch.int64)
            offsets = torch.zeros(batch_size, dtype=torch.int64)  # Example offsets
            indices = torch.arange(num_tokens).expand(batch_size, num_tokens)
            self.input_tensors.append((scores, seeds, offsets, indices))

    def to_cuda(self, input_tensor):
        scores, seeds, offsets, indices = input_tensor
        return (scores.cuda(), seeds.cuda(), offsets.cuda(), indices.cuda())

    def call_op(self, input_tensor):
        scores, seeds, offsets, indices = input_tensor
        return multinomial_sampling(scores, seeds, offsets, indices)

    def get_gbps(self, input_tensor, runtime):
        scores, _, _, _ = input_tensor
        total_bytes = scores.numel() * scores.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        scores, _, _, _ = input_tensor
        FLOPS = 2 * scores.numel()  # Example calculation, adjust based on actual operations
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
