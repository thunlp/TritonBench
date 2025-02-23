import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.int8_matmul_kernel import matmul  # 正确引入算子
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('int8_matmul_kernel', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 33):  # 从小到大取不同 size 的张量
            M = 128 * i
            K = 128 * i  # 注意 K 是 4 的倍数
            N = 128 * i
            a = torch.randint(-128, 127, (M, K), dtype=torch.int8)  # A 是 int8 类型
            b = torch.randint(0, 4, (K // 4, N), dtype=torch.uint8)  # B 是 uint8 类型
            self.input_tensors.append((a, b))

    def to_cuda(self, input_tensor):
        a, b = input_tensor
        return a.cuda(), b.cuda()  # 将张量转移到 CUDA

    def call_op(self, input_tensor):
        a, b = input_tensor
        return matmul(a, b)  # 调用 matmul 算子

    def get_gbps(self, input_tensor, runtime):
        a, b = input_tensor
        M, K = a.shape
        N = b.shape[1]
        # 数据传输量：A (int8) + B (uint8) + C (int32)
        total_bytes = (M * K + K * N // 4 + M * N * 4)  # 注意单位是字节
        GBPS = total_bytes / (runtime / 1000) / 1e9  # 转换为 GB/s
        return GBPS

    def get_tflops(self, input_tensor, runtime):
        A, B = input_tensor
        M, K = A.shape
        K_, N = B.shape
        K_effective = K * 4
        # 计算量：2 * M * N * K
        FLOPS = M * N * K_effective # int8 matmul, FLOPS 2 more than float16 matmul
        TFLOPS = FLOPS / (runtime / 1000) / 1e12  # 转换为 TFLOPS
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
