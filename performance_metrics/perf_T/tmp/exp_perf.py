import sys
import os
import json

sys.path.append('/home/lishangzhan/triton/torch_performance/GPU_efficiency/output_DeepSeek-R1')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exp import exp  # 正确引入exp算子
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('exp', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        # 生成不同size的CPU张量（从2^12到2^27）
        self.input_tensors = []
        for i in range(12, 28):
            size = 2 ** i
            input_tensor = torch.rand(size, dtype=torch.float32)  # 默认使用float32类型
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        # 将输入张量转移到CUDA
        return input_tensor.cuda()
    
    def call_op(self, input_tensor):
        # 调用exp算子
        return exp(input_tensor)
    
    def get_gbps(self, input_tensor, runtime):
        # 计算总数据量（输入+输出）的带宽（GB/s）
        element_size = input_tensor.element_size()  # 获取单个元素字节数
        total_bytes = input_tensor.numel() * element_size * 2  # 输入输出各占一份
        GBPS = total_bytes / (runtime / 1000) / 1e9  # 转换为GB/s（runtime单位ms）
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        # 假设每个元素执行一次指数运算（1 FLOP/element）
        FLOPS = input_tensor.numel()  # 总操作数 = 元素数量
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
