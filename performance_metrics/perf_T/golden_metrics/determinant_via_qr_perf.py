import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.determinant_via_qr import determinant_via_qr
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('determinant_via_qr', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        # 生成不同尺寸的方阵（从16x16到8192x8192）
        for exp in range(4, 14):  # 2^4=16 到 2^13=8192
            n = 2 ** exp
            input_tensor = torch.randn((n, n), dtype=self.dtype)
            self.input_tensors.append(input_tensor)
    
    def to_cuda(self, input_tensor):
        return input_tensor.cuda()
    
    def call_op(self, input_tensor):
        return determinant_via_qr(input_tensor)
    
    def get_gbps(self, input_tensor, runtime):
        # 计算总数据量（输入矩阵+输出标量）
        n = input_tensor.size(0)
        element_size = input_tensor.element_size()
        input_bytes = n * n * element_size
        output_bytes = 1 * element_size  # 标量输出
        total_bytes = input_bytes + output_bytes
        return total_bytes / (runtime / 1000) / 1e9  # GB/s
    
    def get_tflops(self, input_tensor, runtime):
        # 假设主要计算量来源于QR分解（复杂度O(n^3)）
        n = input_tensor.size(0)
        flops = 2 * (n ** 3)  # 假设每个元素需要2次浮点运算
        return flops / (runtime / 1000) / 1e12  # TFLOPS
    def run_benchmark(self):
        results = []
        for input_tensor_ in self.input_tensors:
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
            input_tensor = None
        folder_path = "/home/lishangzhan/triton/torch_performance/results"
        file_name = self.op_name + ".json"
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'w', encoding='utf8') as f:
            json.dump(results, f, indent=4)

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
