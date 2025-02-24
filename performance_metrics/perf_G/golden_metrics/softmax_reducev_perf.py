import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.softmax_reducev import token_softmax_reducev_fwd
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('softmax_reducev', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(5, 12):  # Example sizes, adjust as needed
            batch_size = 2 ** i
            seq_len = 128  # Example sequence length
            d_model = 64  # Example model dimension
            logics = torch.rand((batch_size, seq_len), dtype=torch.float32)
            v = torch.rand((batch_size, seq_len, d_model), dtype=torch.float32)
            o = torch.empty((batch_size, seq_len, d_model), dtype=torch.float32)
            b_loc = torch.randint(0, seq_len, (batch_size, seq_len), dtype=torch.int32)
            b_start_loc = torch.zeros(batch_size, dtype=torch.int32)
            b_seq_len = torch.full((batch_size,), seq_len, dtype=torch.int32)
            max_input_len = seq_len
            other_kv_index = 0  # Example value
            self.input_tensors.append((logics, v, o, b_loc, b_start_loc, b_seq_len, max_input_len, other_kv_index))

    def to_cuda(self, input_tensor):
        # return tuple(tensor.cuda() for tensor in input_tensor)
        logics, v, o, b_loc, b_start_loc, b_seq_len, max_input_len, other_kv_index = input_tensor
        return (logics.cuda(), v.cuda(), o.cuda(), b_loc.cuda(), b_start_loc.cuda(), b_seq_len.cuda(), max_input_len, other_kv_index)

    def call_op(self, input_tensor):
        logics, v, o, b_loc, b_start_loc, b_seq_len, max_input_len, other_kv_index = input_tensor
        return token_softmax_reducev_fwd(logics, v, o, b_loc, b_start_loc, b_seq_len, max_input_len, other_kv_index)

    def get_gbps(self, input_tensor, runtime):
        logics, v, o, _, _, _, _, _ = input_tensor
        total_bytes = (logics.numel() + v.numel() + o.numel()) * logics.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        logics, v, _, _, _, _, _, _ = input_tensor
        FLOPS = 2 * logics.numel() * v.shape[-1]  # Assuming 2 FLOPS per element for softmax and reduce
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
