import sys
import os
import json

sys.path.append('/home/lishangzhan/triton/torch_performance/GPU_efficiency/output_DeepSeek-R1')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fused_hardshrink_dropout import fused_hardshrink_dropout
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_hardshrink_dropout', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):  # 不同规模的数据: 2^12 到 2^27
            size = 2 ** i
            input_tensor = torch.rand(size, dtype=self.dtype)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()  # 迁移到GPU
    
    def call_op(self, input_tensor):
        # 调用融合算子，默认参数: p=0.5, training=True, inplace=False, lambd=0.5
        return fused_hardshrink_dropout(input_tensor, p=0.5, training=True, inplace=False, lambd=0.5)

    def get_gbps(self, input_tensor, runtime):
        # 计算内存带宽: (输入+输出)数据量 / 时间
        total_bytes = input_tensor.numel() * input_tensor.element_size() * 4
        GBPS = total_bytes / (runtime / 1000) / 1e9  # 单位转换为GB/s
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        # 计算计算吞吐量: 每个元素2次浮点操作（Dropout乘法 + Hardshrink绝对值）
        flops = input_tensor.numel() * 2  # 训练模式下每个元素2次操作
        TFLOPS = flops / (runtime / 1000) / 1e12  # 单位转换为TFLOPS
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
