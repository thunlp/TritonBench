import sys
import os
import json

sys.path.append('/home/lishangzhan/triton/torch_performance/GPU_efficiency/output_DeepSeek-R1')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ones_like import ones_like  # 正确引入算子
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('ones_like', dtype=dtype, is_backward=is_backward, **kwargs)  # 修正算子名称

    def get_input_tensors(self):
        self.input_tensors = []
        dtype = self.dtype if self.dtype is not None else torch.float32  # 支持不同数据类型
        for i in range(12, 21):
            size = 2 ** i
            input_tensor = torch.rand(size, dtype=dtype)  # 生成不同大小的CPU张量
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()  # 转移到CUDA

    def call_op(self, input_tensor):
        return ones_like(input_tensor)  # 调用算子

    def get_gbps(self, input_tensor, runtime):
        # 计算总字节数：输入和输出各占一次
        total_bytes = input_tensor.numel() * input_tensor.element_size() * 2
        GBPS = total_bytes / (runtime / 1000) / 1e9  # 转换为GB/s
        return GBPS

    def get_tflops(self, input_tensor, runtime):
        # 假设每个元素需要一次操作（如填充）
        FLOPS = input_tensor.numel()
        TFLOPS = FLOPS / (runtime / 1000) / 1e12  # 转换为TFLOPS
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
