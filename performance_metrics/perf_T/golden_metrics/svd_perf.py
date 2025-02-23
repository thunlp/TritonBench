import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.svd import svd
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('svd', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 9):
            size = 2 ** i
            input_tensor = torch.rand((size, size), dtype=self.dtype)
            self.input_tensors.append(input_tensor)
            
    def to_cuda(self, input_tensor):
        return input_tensor.cuda()
    
    def call_op(self, input_tensor):
        return svd(input_tensor)  # 使用默认full_matrices=True
    
    def get_gbps(self, input_tensor, runtime):
        # 处理不同维度输入（支持batch维度）
        if len(input_tensor.shape) == 2:
            batch, m, n = 1, *input_tensor.shape
        else:
            batch, m, n = input_tensor.shape
        
        element_size = input_tensor.element_size()
        k = min(m, n)
        
        # 输入数据量
        input_bytes = batch * m * n * element_size
        
        # 输出数据量（U: m*m, S: k, Vh: n*n）
        output_bytes = batch * (m*m + n*n + k) * element_size
        
        total_bytes = input_bytes + output_bytes
        return total_bytes / (runtime / 1000) / 1e9  # 转换为GB/s

    def get_tflops(self, input_tensor, runtime):
        # 处理不同维度输入（支持batch维度）
        if len(input_tensor.shape) == 2:
            batch, m, n = 1, *input_tensor.shape
        else:
            batch, m, n = input_tensor.shape
        
        k = min(m, n)
        
        # SVD计算量近似公式（来源：LAPACK文档）
        if m >= n:
            flops_per_matrix = 4 * m * n**2 - (4/3) * n**3
        else:
            flops_per_matrix = 4 * n * m**2 - (4/3) * m**3
        
        total_flops = batch * flops_per_matrix
        return total_flops / (runtime / 1000) / 1e12  # 转换为TFLOPS

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
