import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.fused_rotary_embedding import decoding_fused_rotary_embedding
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_rotary_embedding', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 8):  # Adjust the range as needed for your testing
            total_tokens = 2 ** i
            head_num = 8  # Example head number
            head_dim = 64  # Example head dimension
            kv_head_num = head_num // 2  # Example kv head number
            block_size = 16  # Example block size
            max_position_len = 512  # Example max position length
            bsz = 4  # Example batch size
            max_blocks_per_sequence = 10  # Example max blocks per sequence

            q = torch.rand(total_tokens, head_num, head_dim, dtype=torch.float16)
            k = torch.rand(total_tokens, kv_head_num, head_dim, dtype=torch.float16)
            v = torch.rand(total_tokens, kv_head_num, head_dim, dtype=torch.float16)
            cos = torch.rand(max_position_len, head_dim, dtype=torch.float16)
            sin = torch.rand(max_position_len, head_dim, dtype=torch.float16)
            k_cache = torch.rand(max_blocks_per_sequence, kv_head_num, block_size, head_dim, dtype=torch.float16)
            v_cache = torch.rand(max_blocks_per_sequence, kv_head_num, block_size, head_dim, dtype=torch.float16)
            block_tables = torch.randint(0, max_blocks_per_sequence, (bsz, max_blocks_per_sequence), dtype=torch.int32)
            kv_lengths = torch.randint(1, total_tokens, (bsz,), dtype=torch.int32)

            self.input_tensors.append((q, k, v, cos, sin, k_cache, v_cache, block_tables, kv_lengths))

    def to_cuda(self, input_tensor):
        return tuple(tensor.cuda() for tensor in input_tensor)

    def call_op(self, input_tensor):
        q, k, v, cos, sin, k_cache, v_cache, block_tables, kv_lengths = input_tensor
        return decoding_fused_rotary_embedding(q, k, v, cos, sin, k_cache, v_cache, block_tables, kv_lengths)

    def get_gbps(self, input_tensor, runtime):
        q, k, v, cos, sin, k_cache, v_cache, block_tables, kv_lengths = input_tensor
        total_bytes = (q.numel() + k.numel() + v.numel() + cos.numel() + sin.numel() +
                       k_cache.numel() + v_cache.numel() + block_tables.numel() + kv_lengths.numel()) * q.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        q, k, v, cos, sin, k_cache, v_cache, block_tables, kv_lengths = input_tensor
        # Assuming each element operation is a FLOP
        FLOPS = 2 * (q.numel() + k.numel() + v.numel())
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
