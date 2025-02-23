import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.quantize_dynamic import dynamic_custom
from performance_utils import Performance_Metrics, do_bench_config

import torch
from torch import nn
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('quantize_dynamic', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        # 生成不同大小的线性层模型，测试量化性能
        for i in range(4, 12):  # 可根据需要调整范围
            size = 2 ** i
            model = nn.Linear(size, size)  # 输入输出维度均为size的线性层
            self.input_tensors.append(model)

    def to_cuda(self, model):
        # 动态量化通常在CPU执行，无需移动至CUDA
        return model

    def call_op(self, model):
        # 对模型中的Linear层进行动态量化
        qconfig_spec = {nn.Linear}  # 指定量化所有Linear层
        return dynamic_custom(model, qconfig_spec=qconfig_spec)

    def get_gbps(self, model, runtime):
        # 计算数据传输量：读取原权重 + 写入量化后数据
        weight = model.weight
        M, N = weight.shape
        bytes_read = weight.numel() * weight.element_size()  # 原始数据量（float32）
        bytes_written = weight.numel() * 1 + M * 4  # 量化权重(int8) + 缩放因子(float32)
        total_bytes = bytes_read + bytes_written
        GBPS = total_bytes / (runtime / 1000) / 1e9  # 转换为GB/s
        return GBPS

    def get_tflops(self, model, runtime):
        # 估算浮点操作：每个权重元素3次操作（减、除、取整）
        weight = model.weight
        M, N = weight.shape
        FLOPS = 3 * M * N 
        TFLOPS = FLOPS / (runtime / 1000) / 1e12  # 转换为TFLOPs
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
                "input_size": [input_tensor.weight.shape],
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
