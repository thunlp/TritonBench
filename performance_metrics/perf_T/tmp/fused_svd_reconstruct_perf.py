import sys
import os
import json

sys.path.append('/home/lishangzhan/triton/torch_performance/GPU_efficiency/output_DeepSeek-R1_rag')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fused_svd_reconstruct import fused_svd_reconstruct
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_svd_reconstruct', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        # 生成不同尺寸的二维张量（从64x64到4096x4096）
        for i in range(6, 11):  # 2^6=64 到 2^12=4096
            size = 2 ** i
            input_tensor = torch.rand((size, size), dtype=self.dtype)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()
    
    def call_op(self, input_tensor):
        return fused_svd_reconstruct(input_tensor)
    
    def get_gbps(self, input_tensor, runtime):
        # 输入和输出各一个矩阵，每个矩阵大小相同
        total_bytes = 2 * input_tensor.numel() * input_tensor.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        m, n = input_tensor.shape
        k = min(m, n)
        
        # SVD部分的估算FLOPs（假设为4*m*n*k）
        svd_flops = 4 * m * n * k
        
        # 矩阵乘法部分的FLOPs
        matmul_flops = 2 * k * n * (k + m)  # 2*k^2*n + 2*m*k*n
        
        total_flops = svd_flops + matmul_flops
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS

    def run_benchmark(self):
        results = []
        for input_tensor_ in self.input_tensors:
            try:
                input_tensor = self.to_cuda(input_tensor_)
                # print(input_tensor)
                op = lambda : self.call_op(input_tensor)
                ms = self.get_runtime(op)
                gbps = self.get_gbps(input_tensor, ms)
                tflops = self.get_tflops(input_tensor, ms)
                result = {
                    "input_size": [input_tensor.shape],
                    "ms": ms,
                    "GB/s": gbps,
                    "TFLOPS": tflops
                }
                print(result)
                results.append(result)
            except Exception as e:
                print(f"Failed to run benchmark for input tensor. Error: {e}")
            input_tensor = None
        folder_path = "./results"
        file_name = self.op_name + ".json"
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'w', encoding='utf8') as f:
            json.dump(results, f, indent=4)

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
