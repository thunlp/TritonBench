import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.rms_matmul_rbe import rms_matmul_rbe_qkv_wrapper
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('rms_matmul_rbe', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 12):  # Adjust the range as needed for your testing
            batch_size = 32
            M = 2 ** i
            K = 2 ** (i - 1)
            N = 2 ** (i - 1)
            n_heads = 8
            head_dim = N // n_heads

            x = torch.rand((batch_size, M, K), dtype=torch.float16)
            q_weight = torch.rand((K, N), dtype=torch.float16)
            k_weight = torch.rand((K, N), dtype=torch.float16)
            v_weight = torch.rand((K, N), dtype=torch.float16)
            rms_w = torch.rand((K,), dtype=torch.float16)
            k = torch.rand((batch_size, M, N), dtype=torch.float16)
            v = torch.rand((batch_size, M, N), dtype=torch.float16)

            self.input_tensors.append((x, 0, q_weight, k_weight, v_weight, rms_w, n_heads, head_dim, k, v))

    def to_cuda(self, input_tensor):
        return tuple(tensor.cuda() if isinstance(tensor, torch.Tensor) else tensor for tensor in input_tensor)

    def call_op(self, input_tensor):
        return rms_matmul_rbe_qkv_wrapper(*input_tensor)

    def get_gbps(self, input_tensor, runtime):
        x, _, q_weight, k_weight, v_weight, rms_w, _, _, k, v = input_tensor
        total_bytes = (x.numel() + q_weight.numel() + k_weight.numel() + v_weight.numel() + rms_w.numel() + k.numel() + v.numel()) * x.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        x, _, q_weight, _, _, _, _, _, _, _ = input_tensor
        batch, M, K = x.shape
        N = q_weight.shape[1]
        FLOPS = 2 * batch * M * N * K  # Assuming 2 operations per multiply-add
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
