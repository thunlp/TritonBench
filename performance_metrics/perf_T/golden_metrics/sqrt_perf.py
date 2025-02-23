import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.sqrt import sqrt  # 正确引入sqrt算子
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('sqrt', dtype=dtype, is_backward=is_backward, **kwargs)  # 修正为算子名称

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):  # 不同规模的输入（2^12到2^27元素）
            size = 2 ** i
            input_tensor = torch.rand(size, dtype=self.dtype)  # 根据dtype参数生成张量
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()  # 张量迁移到CUDA
    
    def call_op(self, input_tensor):
        return sqrt(input_tensor)  # 调用sqrt算子
    
    def get_gbps(self, input_tensor, runtime):
        # 计算内存带宽：输入输出各占一份数据量
        total_bytes = input_tensor.numel() * input_tensor.element_size() * 2
        GBPS = total_bytes / (runtime / 1000) / 1e9  # 转换为GB/s
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        # 计算计算量：每个元素一次平方根操作
        FLOPS = input_tensor.numel()
        TFLOPS = FLOPS / (runtime / 1000) / 1e12  # 转换为TFLOPS
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
