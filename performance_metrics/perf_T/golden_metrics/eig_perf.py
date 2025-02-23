import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.eig import eig  # 正确引入特征值分解算子
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('eig', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        # 生成不同尺寸的方阵 (2^6 到 2^10)
        for i in range(6, 11):
            n = 2 ** i
            input_tensor = torch.rand(n, n, dtype=self.dtype)  # 实数方阵
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()  # 张量迁移到GPU
    
    def call_op(self, input_tensor):
        return eig(input_tensor)  # 调用特征值分解
    
    def get_gbps(self, input_tensor, runtime):
        """ 计算内存吞吐量 (GB/s) """
        n = input_tensor.shape[0]
        dtype = input_tensor.dtype
        
        # 确定复数元素字节大小
        complex_element_size = 8 if dtype == torch.float32 else 16
        
        # 输入矩阵的字节数
        input_bytes = input_tensor.numel() * input_tensor.element_size()
        # 输出特征值字节数 (n个复数)
        eigenvalues_bytes = n * complex_element_size
        # 输出特征向量字节数 (n*n复数)
        eigenvectors_bytes = n * n * complex_element_size
        
        total_bytes = input_bytes + eigenvalues_bytes + eigenvectors_bytes
        return total_bytes / (runtime / 1000) / 1e9  # 转换为GB/s

    def get_tflops(self, input_tensor, runtime):
        """ 估算浮点运算量 (TFLOPS) """
        n = input_tensor.shape[0]
        # 假设特征值分解需要约25n³次浮点运算
        flops_estimate = 25 * (n ** 3)
        return flops_estimate / (runtime / 1000) / 1e12  # 转换为TFLOPS

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
