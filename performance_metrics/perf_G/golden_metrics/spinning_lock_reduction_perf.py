import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.spinning_lock_reduction import spinning_lock
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('spinning_lock_reduction', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 12):  # Example sizes, adjust as needed
            print(i)
            M = N = 2 ** i
            k = 4  # Example value
            BLOCK_SIZE_M = 128
            BLOCK_SIZE_N = 128
            num_sms = 108  # Example value, adjust based on your GPU
            P = torch.zeros((num_sms * BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=torch.float16)
            C = torch.zeros((M, N), dtype=torch.float16)
            stride_cm = C.stride(0)
            stride_cn = C.stride(1)
            locks = torch.zeros(M, dtype=torch.int32)           
            
            self.input_tensors.append((P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N))

    def to_cuda(self, input_tensor):
        P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N = input_tensor
        return (P.cuda(), C.cuda(), locks.cuda(), num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N)

    def call_op(self, input_tensor):
        P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N = input_tensor
        spinning_lock(P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N)
        return C

    def get_gbps(self, input_tensor, runtime):
        P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N = input_tensor
        total_bytes = num_sms / k * 9 * (k - 1) * BLOCK_SIZE_M * BLOCK_SIZE_N * 2 + (num_sms - num_sms / k) * 9 * BLOCK_SIZE_M * BLOCK_SIZE_N * 2 + num_sms * 9 * BLOCK_SIZE_M * BLOCK_SIZE_N * 2
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N = input_tensor
        # Assuming each element in P contributes to a FLOP
        FLOPS = 9 * num_sms * (k - 1) * BLOCK_SIZE_M * BLOCK_SIZE_N / k  # Example calculation, adjust based on actual computation
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
