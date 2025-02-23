import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.context_attn_mistral import context_attention_fwd
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('context_attn_mistral', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 14):  # Adjust the range as needed for different sizes
            print(i)
            Z, H, D_HEAD = 4, 16, 128
            N_CTX = 2 ** i
            dtype = torch.float16
            q = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
            k = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.4, std=0.2)
            v = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)
            o = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)
            max_input_len = Z * N_CTX
            b_start_loc = torch.zeros((Z,), dtype=torch.int32)
            b_seq_len = torch.full((Z,), N_CTX, dtype=torch.int32)
            for i in range(1, Z):
                b_start_loc[i] = b_start_loc[i - 1] + b_seq_len[i - 1]
            # b_prompt_cache_len = torch.full((Z,), prompt_cache_len, dtype=torch.int32)

            self.input_tensors.append((q, k, v, o, b_start_loc, b_seq_len, max_input_len, 512))

    def to_cuda(self, input_tensor):
        # return tuple(tensor.cuda() for tensor in input_tensor)
        q, k, v, o, b_start_loc, b_seq_len, max_input_len, sliding_window = input_tensor
        return tuple([q.cuda(), k.cuda(), v.cuda(), o.cuda(), b_start_loc.cuda(), b_seq_len.cuda(), max_input_len, sliding_window])

    def call_op(self, input_tensor):
        q, k, v, o, b_start_loc, b_seq_len, max_input_len, sliding_window = input_tensor
        context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_input_len, sliding_window)
        return o

    def get_gbps(self, input_tensor, runtime):
        q, k, v, o, _, _, _, _ = input_tensor
        total_bytes = (q.numel() + k.numel() + v.numel() + o.numel()) * q.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        q, _, _, _, _, _, _, _ = input_tensor
        seq_len, head, d_model = q.shape
        # Assuming FLOPS is calculated as 2 * batch_size * head * seq_len * seq_len * d_model
        FLOPS = 2 * head * seq_len * d_model
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
