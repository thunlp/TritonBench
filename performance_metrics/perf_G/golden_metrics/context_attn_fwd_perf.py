import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.context_attn_fwd import context_attention_fwd_ppl_int8kv
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('context_attn_fwd', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 14):  # Adjust the range as needed for your testing
            Z, H, D_HEAD = 4, 16, 256
            N_CTX = 2 ** i
            dtype = torch.float16
            prompt_cache_len = 256
            q = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype).normal_(mean=0.1, std=0.2)
            kv = torch.empty((Z, 2 * H, N_CTX, D_HEAD), dtype=dtype).normal_(mean=0.4, std=0.2)
            k = kv[:, :H]
            v = kv[:, H:]
            o = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype).normal_(mean=0.3, std=0.2)
            max_input_len = Z * N_CTX
            b_start_loc = torch.zeros((Z,), dtype=torch.int32)
            b_seq_len = torch.ones((Z,), dtype=torch.int32)
            b_prompt_cache_len = torch.full((Z,), prompt_cache_len, dtype=torch.int32)

            self.input_tensors.append((q, k, v, o, b_start_loc, b_seq_len, max_input_len, b_prompt_cache_len))

    def to_cuda(self, input_tensor):
        # return tuple(tensor.cuda() for tensor in input_tensor)
        q, k, v, o, b_start_loc, b_seq_len, max_input_len, b_prompt_cache_len = input_tensor
        return tuple([q.cuda(), k.cuda(), v.cuda(), o.cuda(), b_start_loc.cuda(), b_seq_len.cuda(), max_input_len, b_prompt_cache_len.cuda()])

    def call_op(self, input_tensor):
        q, k, v, o, b_start_loc, b_seq_len, max_input_len, b_prompt_cache_len = input_tensor
        context_attention_fwd_ppl_int8kv(q, k, v, o, b_start_loc, b_seq_len, max_input_len, b_prompt_cache_len)
        return o

    def get_gbps(self, input_tensor, runtime):
        q, k, v, o, b_start_loc, b_seq_len, max_input_len, b_prompt_cache_len = input_tensor
        total_bytes = (q.numel() + k.numel() + v.numel() + o.numel()) * q.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        q, k, v, o, b_start_loc, b_seq_len, max_input_len, b_prompt_cache_len = input_tensor
        seq_len, head, d_model = q.shape
        FLOPS = 2 * head * seq_len * d_model
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
