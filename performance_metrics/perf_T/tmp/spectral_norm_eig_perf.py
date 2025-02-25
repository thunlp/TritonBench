import sys
import os
import json

sys.path.append('/home/lishangzhan/triton/torch_performance/GPU_efficiency/output_DeepSeek-R1_rag')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spectral_norm_eig import spectral_norm_eig
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('spectral_norm_eig', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        # 生成不同大小的方阵，如16x16, 32x32, ..., 512x512
        for i in range(2, 8):  # 可根据需要调整i的范围
            n = 2 ** i
            input_tensor = torch.rand(n, n, dtype=self.dtype or torch.float32)
            self.input_tensors.append(input_tensor)
    
    def to_cuda(self, input_tensor):
        return input_tensor.cuda()
        
    def call_op(self, input_tensor):
        return spectral_norm_eig(input_tensor)
    
    def get_gbps(self, input_tensor, runtime):
        # 输入和输出的总字节数
        n = input_tensor.shape[-1]
        batch_numel = input_tensor.numel() // (n * n)  # 批处理维度的大小
        input_bytes = input_tensor.numel() * input_tensor.element_size()
        output_bytes = batch_numel * input_tensor.element_size()  # 输出为每个矩阵的谱范数
        total_bytes = input_bytes + output_bytes
        GBPS = total_bytes / (runtime / 1000) / 1e9  # 转换为GB/s
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        # 假设特征值分解的浮点运算量为 ~ (8/3) * n^3 次，每个矩阵
        n = input_tensor.shape[-1]
        batch_numel = input_tensor.numel() // (n * n)  # 批处理维度的大小
        flops_per_matrix = (8/3) * n ** 3  # 根据算法复杂度估算
        total_flops = batch_numel * flops_per_matrix
        TFLOPS = total_flops / (runtime / 1000) / 1e12  # 转换为TFLOPS
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
