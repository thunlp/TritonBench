import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.ldl_factor import ldl_factor
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('ldl_factor', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        # 生成不同尺寸的对称正定矩阵（从32x32到4096x4096）
        for i in range(5, 11):  # 可根据硬件调整范围
            n = 2 ** i
            # 生成对称正定矩阵
            A = torch.rand(n, n, dtype=self.dtype or torch.float32)
            A = A + A.T  # 确保对称性
            A += n * torch.eye(n, dtype=self.dtype or torch.float32)  # 对角增强保证正定
            self.input_tensors.append(A)
    
    def to_cuda(self, input_tensor):
        return input_tensor.cuda()
    
    def call_op(self, input_tensor):
        # 执行LDL分解并同步CUDA确保准确计时
        result = ldl_factor(input_tensor)
        torch.cuda.synchronize()  # 确保CUDA操作完成
        return result
    
    def get_gbps(self, input_tensor, runtime):
        n = input_tensor.size(0)
        element_size = input_tensor.element_size()
        # 总数据量 = 输入矩阵 + 输出LD矩阵 + 输出pivots
        total_bytes = (n*n*element_size) * 2 + (n * 8)  # 输入+LD各n²，pivots为int64
        GBPS = total_bytes / (runtime / 1000) / 1e9  # 转换为GB/s
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        n = input_tensor.size(0)
        flops = n**3 / 3  # LDL分解计算量约为n³/3次浮点运算
        TFLOPS = flops / (runtime / 1000) / 1e12  # 转换为TFLOPS
        return TFLOPS
    
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
