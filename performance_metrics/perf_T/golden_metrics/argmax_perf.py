import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.argmax import argmax
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, dim=0, **kwargs):
        super().__init__('argmax', dtype=dtype, is_backward=is_backward, **kwargs)
        self.dim = dim  # 新增dim参数用于指定计算维度

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 30):
            size = 2 ** i
            # 生成一维测试张量（可修改为更高维度）
            input_tensor = torch.rand(size, dtype=torch.float16)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()
    
    def call_op(self, input_tensor):
        # 调用时使用初始化时指定的dim参数
        return argmax(input_tensor, dim=self.dim, keepdim=False)
    
    def get_gbps(self, input_tensor, runtime):
        # 输入数据量（考虑输入输出数据类型差异）
        input_bytes = input_tensor.numel() * input_tensor.element_size()
        # 输出数据量（argmax返回int64类型）
        output_numel = input_tensor.numel() // input_tensor.shape[self.dim]
        output_bytes = output_numel * 8
        # 总数据量计算
        total_bytes = input_bytes + output_bytes
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        # 假设每个元素需要一次比较操作
        FLOPS = input_tensor.numel()
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
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
