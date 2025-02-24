import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.context_attn_llama import context_attention_fwd
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('context_attn_llama', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 12):  # Adjust the range as needed for your testing
            # batch_size = 32
            # seq_len = 2 ** i  # Example sequence length
            # head_dim = 64  # Example head dimension
            # num_heads = 8  # Example number of heads

            # q = torch.rand((batch_size, num_heads, seq_len, head_dim), dtype=torch.float16)
            # k = torch.rand((batch_size, num_heads, seq_len, head_dim), dtype=torch.float16)
            # v = torch.rand((batch_size, num_heads, seq_len, head_dim), dtype=torch.float16)
            # o = torch.empty_like(q)
            # b_req_idx = torch.zeros(batch_size, dtype=torch.int32)
            # b_start_loc = torch.arange(0, batch_size * seq_len, seq_len, dtype=torch.int32)
            # b_seq_len = torch.full((batch_size,), seq_len, dtype=torch.int32)
            # b_prompt_cache_len = torch.zeros(batch_size, dtype=torch.int32)
            # max_input_len = seq_len
            # req_to_token_indexs = torch.arange(seq_len, dtype=torch.int32).repeat(batch_size, 1)
            print(i)
            Z, H, D_HEAD = 4, 16, 256
            N_CTX = 2 ** i
            dtype = torch.float16
            prompt_cache_len = 256
            q = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype).normal_(mean=0.1, std=0.2)
            k = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype).normal_(mean=0.4, std=0.2)
            v = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype).normal_(mean=0.3, std=0.2)
            o = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype).normal_(mean=0.3, std=0.2)

            req_to_token_indexs = torch.empty((1000, N_CTX), dtype=torch.int32)
            max_input_len = Z * N_CTX
            b_start_loc = torch.zeros((Z,), dtype=torch.int32)
            # b_seq_len = torch.ones((Z,), dtype=torch.int32)
            b_seq_len = torch.full((Z,), N_CTX, dtype=torch.int32)
            b_req_idx = torch.ones((Z,), dtype=torch.int32)
            # b_prompt_cache_len = torch.zeros(Z, dtype=torch.int32)
            b_prompt_cache_len = torch.full((Z,), prompt_cache_len, dtype=torch.int32)

            self.input_tensors.append((q, k, v, o, b_req_idx, b_start_loc, b_seq_len, b_prompt_cache_len, max_input_len, req_to_token_indexs))

    def to_cuda(self, input_tensor):
        # return tuple(tensor.cuda() for tensor in input_tensor)
        q, k, v, o, b_req_idx, b_start_loc, b_seq_len, b_prompt_cache_len, max_input_len, req_to_token_indexs = input_tensor
        return tuple([q.cuda(), k.cuda(), v.cuda(), o.cuda(), b_req_idx.cuda(), b_start_loc.cuda(), b_seq_len.cuda(), b_prompt_cache_len.cuda(), max_input_len, req_to_token_indexs.cuda()])

    def call_op(self, input_tensor):
        q, k, v, o, b_req_idx, b_start_loc, b_seq_len, b_prompt_cache_len, max_input_len, req_to_token_indexs = input_tensor
        context_attention_fwd(q, k, v, o, b_req_idx, b_start_loc, b_seq_len, b_prompt_cache_len, max_input_len, req_to_token_indexs)
        return o

    def get_gbps(self, input_tensor, runtime):
        q, _, _, _, _, _, _, _, _, _ = input_tensor
        total_bytes = 4 * q.numel() * q.element_size()  # q, k, v, and o are all involved
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        q, _, _, _, _, _, _, _, _, _ = input_tensor
        FLOPS = 2 * q.size(0) * q.size(1) * q.size(2)  # Simplified FLOPS calculation
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
