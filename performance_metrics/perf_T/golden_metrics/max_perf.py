import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.max import max
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('max', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):  # 测试不同大小的输入
            size = 2 ** i
            dtype = self.dtype if self.dtype is not None else torch.float32
            input_tensor = torch.rand(size, dtype=dtype)
            self.input_tensors.append(input_tensor)
    
    def to_cuda(self, input_tensor):
        input_tensor = input_tensor.cuda()
        return input_tensor
        
    def call_op(self, input_tensor):
        values, indices = max(input_tensor, dim=0)  # 沿dim=0取最大值
        return (values, indices)  # 前向测试返回元组
    
    def get_gbps(self, input_tensor, runtime):
        # 总数据量 = 输入数据 + 输出values数据 + 输出indices数据
        input_bytes = input_tensor.numel() * input_tensor.element_size()
        values_bytes = 1 * input_tensor.element_size()  # values为标量
        indices_bytes = 1 * 8  # indices为int64类型，占8字节
        total_bytes = input_bytes + values_bytes + indices_bytes
        GBPS = total_bytes / (runtime / 1000) / 1e9  # 转换为GB/s
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        flops = input_tensor.numel()  # 假设每个元素参与一次比较操作
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
