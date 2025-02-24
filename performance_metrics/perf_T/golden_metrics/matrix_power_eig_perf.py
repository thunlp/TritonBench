import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.matrix_power_eig import matrix_power_eig
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, k=2, **kwargs):
        super().__init__('matrix_power_eig', dtype=dtype, is_backward=is_backward, **kwargs)
        self.k = k  # 幂次参数

    def get_input_tensors(self):
        """生成不同尺寸的对称方阵（保证特征分解稳定性）"""
        self.input_tensors = []
        for i in range(2, 14):
            n = 128 * i
            # 生成对称矩阵（实对称矩阵保证特征值为实数）
            A = torch.randn(n, n, dtype=self.dtype or torch.float32)
            A = (A + A.mT) / 2  # 强制对称
            self.input_tensors.append(A)

    def to_cuda(self, input_tensor):
        """张量转移到CUDA"""
        return input_tensor.cuda()

    def call_op(self, input_tensor):
        """执行矩阵幂运算"""
        return matrix_power_eig(input_tensor, self.k)

    def get_gbps(self, input_tensor, runtime):
        """计算内存带宽（GB/s）"""
        # 输入输出总数据量：输入矩阵 + 输出矩阵
        total_bytes = input_tensor.numel() * input_tensor.element_size() * 2
        return total_bytes / (runtime / 1000) / 1e9  # 转换为GB/s

    def get_tflops(self, input_tensor, runtime):
        """计算浮点性能（TFLOPS）"""
        n = input_tensor.size(-1)  # 矩阵维度
        
        # FLOPS估算（特征分解 + 两次矩阵乘法）
        # 特征分解: ~4/3 n^3 (实对称矩阵eigh的复杂度)
        # 矩阵乘法: 2 * 2n^3 = 4n^3 (两次矩阵乘法)
        flops = (4/3 + 4) * (n ** 3)
        
        return flops / (runtime / 1000) / 1e12  # 转换为TFLOPS
    
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
