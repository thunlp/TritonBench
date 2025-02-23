import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.apply_penalty import apply_penalty
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('apply_penalty', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 10):  # Adjust the range based on expected input sizes
            batch_size = 128
            seq_len = 2 ** i  # Example sequence length
            vocab_size = 30522  # Example vocabulary size

            Logits = torch.rand((batch_size, vocab_size), dtype=torch.float16)
            presence_penalty = torch.rand(batch_size, dtype=torch.float16)
            freqency_penalty = torch.rand(batch_size, dtype=torch.float16)
            repetition_penalty = torch.rand(batch_size, dtype=torch.float16)
            p_token_ids = torch.randint(0, vocab_size, (batch_size * seq_len,), dtype=torch.int32)
            p_token_counts = torch.randint(1, 10, (batch_size * seq_len,), dtype=torch.int32)
            p_cumsum_seq_len = torch.cumsum(torch.randint(1, seq_len, (batch_size + 1,), dtype=torch.int32), dim=0)
            p_max_len_in_batch = seq_len

            self.input_tensors.append((Logits, presence_penalty, freqency_penalty, repetition_penalty,
                                       p_token_ids, p_token_counts, p_cumsum_seq_len, p_max_len_in_batch))

    def to_cuda(self, input_tensor):
        # return tuple(tensor.cuda() for tensor in input_tensor)
        Logits, presence_penalty, freqency_penalty, repetition_penalty, p_token_ids, p_token_counts, p_cumsum_seq_len, p_max_len_in_batch = input_tensor
        return (Logits.cuda(), presence_penalty.cuda(), freqency_penalty.cuda(), repetition_penalty.cuda(), p_token_ids.cuda(), p_token_counts.cuda(), p_cumsum_seq_len.cuda(), p_max_len_in_batch)

    def call_op(self, input_tensor):
        return apply_penalty(*input_tensor)

    def get_gbps(self, input_tensor, runtime):
        Logits = input_tensor[0]
        total_bytes = Logits.numel() * Logits.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        Logits = input_tensor[0]
        FLOPS = 2 * Logits.numel()  # Assuming 2 operations per element
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
