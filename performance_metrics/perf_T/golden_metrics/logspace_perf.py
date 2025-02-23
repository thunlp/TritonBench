import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.logspace import logspace
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('logspace', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        # 生成不同steps大小的测试用例 (2^12 到 2^28)
        for i in range(12, 28):
            steps = 2 ** i
            self.input_tensors.append(steps)

    def to_cuda(self, input_steps):
        # 参数为标量无需转移，直接返回
        return input_steps

    def call_op(self, steps):
        # 调用logspace生成指定steps的张量
        return logspace(
            start=0.0, 
            end=10.0, 
            steps=steps, 
            dtype=self.dtype, 
            device="cuda"
        )

    def get_gbps(self, steps, runtime):
        # 计算GBPS：输出张量总字节数 / 时间
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        total_bytes = steps * element_size  # 输出张量大小
        gbps = total_bytes / (runtime / 1000) / 1e9  # 转换为GB/s
        return gbps

    def get_tflops(self, steps, runtime):
        # 假设每个元素需一次指数运算（实际计算量可能更高）
        flops = steps  # 总操作数
        tflops = flops / (runtime / 1000) / 1e12  # 转换为TFLOPS
        return tflops
    
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
                "input_size": [input_tensor],
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
