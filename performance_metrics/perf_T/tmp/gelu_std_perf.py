import sys
import os
import json

sys.path.append('/home/lishangzhan/triton/torch_performance/GPU_efficiency/output_DeepSeek-R1')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gelu_std import gelu_std
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('gelu_std', dtype=dtype, is_backward=is_backward, **kwargs)
        self.output_sizes = []  # 保存每个输入张量对应的输出元素数目

    def get_input_tensors(self):
        self.input_tensors = []
        self.output_sizes = []
        # 生成不同大小的输入张量，从2^12到2^27
        for i in range(12, 28):
            size = 2 ** i
            input_tensor = torch.rand(size, dtype=torch.float32)
            # 预先计算输出大小并保存
            output_tensor = self.call_op(input_tensor)
            self.input_tensors.append(input_tensor)
            self.output_sizes.append(output_tensor.numel())

    def to_cuda(self, input_tensor):
        # 将输入张量转移到CUDA
        return input_tensor.cuda()
    
    def call_op(self, input_tensor):
        # 调用gelu_std算子，使用默认参数（dim=None，输出标量）
        return gelu_std(input_tensor)
    
    def get_gbps(self, input_tensor, runtime):
        # 根据输入张量的形状查找对应的输出元素数目
        shape = input_tensor.shape
        index = None
        for i, tensor in enumerate(self.input_tensors):
            if tensor.shape == shape:
                index = i
                break
        if index is None:
            raise ValueError("Input tensor shape not found in precomputed list.")
        output_size = self.output_sizes[index]
        # 总字节数 = 输入大小 + 输出大小
        total_bytes = (input_tensor.numel() + output_size) * input_tensor.element_size() * 2
        # 转换为GB/s
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        # 假设每个元素的计算量为11 FLOPs（GELU:8 + std:3）
        flops_per_element = 11
        total_flops = input_tensor.numel() * flops_per_element
        # 转换为TFLOPS
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
