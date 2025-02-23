import sys
import os
import json

sys.path.append('/home/lishangzhan/triton/torch_performance/GPU_efficiency/output_DeepSeek-R1')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from invert_matrix_lu import invert_matrix_lu
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('invert_matrix_lu', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        # 生成不同尺寸的方阵（n从2^6到2^12）
        for i in range(6, 12):
            n = 2 ** i
            # 生成随机可逆矩阵：随机矩阵 + 单位矩阵确保非奇异
            A = torch.rand((n, n), dtype=self.dtype) + torch.eye(n, dtype=self.dtype) * 1e-3
            self.input_tensors.append(A)
    
    def to_cuda(self, input_tensor):
        return input_tensor.cuda()
        
    def call_op(self, input_tensor):
        return invert_matrix_lu(input_tensor)
    
    def get_gbps(self, input_tensor, runtime):
        n = input_tensor.size(0)
        element_size = input_tensor.element_size()
        # 总数据传输量：输入和输出各n^2个元素
        total_bytes = 2 * n * n * element_size
        GBPS = total_bytes / (runtime / 1000) / 1e9  # 转换为GB/s
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        n = input_tensor.size(0)
        # 计算总FLOPS（假设为8/3 * n^3，基于LU分解和解方程步骤）
        flops = (8.0 / 3.0) * n ** 3
        TFLOPS = flops / (runtime / 1000) / 1e12  # 转换为TFLOPS
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
