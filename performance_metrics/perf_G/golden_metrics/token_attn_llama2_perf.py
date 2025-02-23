import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.token_attn_llama2 import token_att_fwd
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('token_attn_llama2', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(5, 16):  # Example sizes, adjust as needed
            batch_size = 2 ** i
            head_num = 8
            seq_len = 128
            d_model = 64
            max_input_len = seq_len

            q = torch.rand((batch_size, head_num, d_model), dtype=torch.float16)
            k = torch.rand((batch_size, head_num, d_model), dtype=torch.float16)
            att_out = torch.zeros((batch_size, head_num, seq_len), dtype=torch.float16)
            B_Loc = torch.randint(0, seq_len, (batch_size, seq_len), dtype=torch.int32)
            B_Start_Loc = torch.zeros(batch_size, dtype=torch.int32)
            B_Seqlen = torch.full((batch_size,), seq_len, dtype=torch.int32)

            self.input_tensors.append((q, k, att_out, B_Loc, B_Start_Loc, B_Seqlen, max_input_len))

    def to_cuda(self, input_tensor):
        q, k, att_out, B_Loc, B_Start_Loc, B_Seqlen, max_input_len = input_tensor
        return (q.cuda(), k.cuda(), att_out.cuda(), B_Loc.cuda(), B_Start_Loc.cuda(), B_Seqlen.cuda(), max_input_len)

    def call_op(self, input_tensor):
        q, k, att_out, B_Loc, B_Start_Loc, B_Seqlen, max_input_len = input_tensor
        return token_att_fwd(q, k, att_out, B_Loc, B_Start_Loc, B_Seqlen, max_input_len)

    def get_gbps(self, input_tensor, runtime):
        q, k, att_out, B_Loc, B_Start_Loc, B_Seqlen, max_input_len = input_tensor
        total_bytes = (q.numel() + k.numel() + att_out.numel()) * q.element_size() + B_Loc.numel() * B_Loc.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        q, k, att_out, B_Loc, B_Start_Loc, B_Seqlen, max_input_len = input_tensor
        # Assuming the main computation is the dot product in the attention mechanism
        FLOPS = 2 * q.size(0) * q.size(1) * q.size(2) * max_input_len
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
