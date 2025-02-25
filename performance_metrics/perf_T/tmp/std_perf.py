import sys
import os
import json

sys.path.append('/home/lishangzhan/triton/torch_performance/GPU_efficiency/output_DeepSeek-R1_rag')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from std import std
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, dim=None, correction=1, keepdim=False, **kwargs):
        super().__init__('std', dtype=dtype, is_backward=is_backward, **kwargs)
        self.dim = dim
        self.correction = correction
        self.keepdim = keepdim
        self.output_sizes = []  # 存储每个输入对应的输出元素数

    def get_input_tensors(self):
        self.input_tensors = []
        self.output_sizes = []
        # 生成不同大小的输入张量（从2^12到2^27）
        for i in range(12, 28):
            size = 2 ** i
            input_tensor = torch.rand(size, dtype=self.dtype or torch.float32)
            self.input_tensors.append(input_tensor)
            # 预计算输出大小用于GBPS计算
            # with torch.no_grad():
            #     output = std(input_tensor, dim=self.dim, 
            #                 correction=self.correction, 
            #                 keepdim=self.keepdim)
            #     self.output_sizes.append(output.numel())

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()
    
    def call_op(self, input_tensor):
        return std(input_tensor, 
                  dim=self.dim,
                  correction=self.correction,
                  keepdim=self.keepdim)
    
    def get_gbps(self, input_tensor, runtime):
        output_size = std(input_tensor, dim=self.dim, correction=self.correction, keepdim=self.keepdim).numel()
        element_size = input_tensor.element_size()
        total_bytes = (input_tensor.numel() + output_size) * element_size
        return total_bytes / (runtime / 1000) / 1e9  # 转换为GB/s
    
    def get_tflops(self, input_tensor, runtime):
        # 计算浮点操作量（3次操作/元素：均值+方差+开方）
        flops = 3 * input_tensor.numel()
        return flops / (runtime / 1000) / 1e12  # 转换为TFLOP/s
    
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
