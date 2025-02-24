import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.token_attn_mistral import token_att_fwd2
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('token_attn_mistral', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 16):  # Adjust the range as needed for your testing
            batch_size = 2 ** i
            seq_len = 128
            head = 8  # Example head size
            dim = 64  # Example dimension size
            sliding_window = 64

            prob = torch.rand((head, seq_len), dtype=torch.float16)
            v = torch.rand((head, seq_len, dim), dtype=torch.float16)
            out = torch.zeros((batch_size, head, dim), dtype=torch.float16)
            Req_to_tokens = torch.randint(0, seq_len, (batch_size, seq_len), dtype=torch.int32)
            B_req_idx = torch.randint(0, batch_size, (batch_size,), dtype=torch.int32)
            B_Start_Loc = torch.zeros((batch_size,), dtype=torch.int32)
            B_Seqlen = torch.full((batch_size,), seq_len, dtype=torch.int32)
            B_Att_Start_Loc = torch.zeros((batch_size,), dtype=torch.int32)
            B_Att_Seqlen = torch.full((batch_size,), seq_len, dtype=torch.int32)

            self.input_tensors.append((prob, v, out, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen, B_Att_Start_Loc, B_Att_Seqlen, sliding_window))

    def to_cuda(self, input_tensor):
        # return tuple(tensor.cuda() for tensor in input_tensor)
        prob, v, out, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen, B_Att_Start_Loc, B_Att_Seqlen, sliding_window = input_tensor
        return (prob.cuda(), v.cuda(), out.cuda(), Req_to_tokens.cuda(), B_req_idx.cuda(), B_Start_Loc.cuda(), B_Seqlen.cuda(), B_Att_Start_Loc.cuda(), B_Att_Seqlen.cuda(), sliding_window)

    def call_op(self, input_tensor):
        prob, v, out, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen, B_Att_Start_Loc, B_Att_Seqlen, sliding_window = input_tensor
        return token_att_fwd2(prob, v, out, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen, B_Att_Start_Loc, B_Att_Seqlen, sliding_window)

    def get_gbps(self, input_tensor, runtime):
        prob, v, out, _, _, _, _, _, _, _ = input_tensor
        total_bytes = (prob.numel() + v.numel() + out.numel()) * prob.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        prob, v, _, _, _, _, _, _, _, _ = input_tensor
        FLOPS = 2 * prob.numel() * v.size(-1)  # Assuming each element in prob contributes to a FLOP
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
