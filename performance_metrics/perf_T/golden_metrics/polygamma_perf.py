import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.polygamma import polygamma  # 正确引入算子
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=torch.float32, is_backward=False, n=1, **kwargs):
        super().__init__('polygamma', dtype=dtype, is_backward=is_backward, **kwargs)
        self.n = n  # 支持不同阶数的测试

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):  # 2^12到2^28的不同大小
            size = 2 ** i
            # 生成正数输入（polygamma要求输入>0）
            input_tensor = torch.rand(size, dtype=self.dtype) + 0.1
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()  # 转移到CUDA设备
    
    def call_op(self, input_tensor):
        return polygamma(self.n, input_tensor)  # 算子调用方式
    
    def get_gbps(self, input_tensor, runtime):
        # 计算内存带宽：输入输出总数据量 / 时间
        total_bytes = input_tensor.numel() * input_tensor.element_size() * 4
        return total_bytes / (runtime / 1000) / 1e9
    
    def get_tflops(self, input_tensor, runtime):
        # 假设每个元素需要n+1次浮点运算（根据阶数不同调整）
        flops = input_tensor.numel() * (self.n + 1)
        return flops / (runtime / 1000) / 1e12
    
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
