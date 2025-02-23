import sys
import os
import json
import math

sys.path.append('/home/lishangzhan/triton/torch_performance/GPU_efficiency/output_DeepSeek-R1')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fftn import fftn
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fftn', dtype=dtype, is_backward=is_backward, **kwargs)
        self.s = kwargs.get('s', None)
        self.dim = kwargs.get('dim', None)
        self.norm = kwargs.get('norm', None)

    def get_input_tensors(self):
        self.input_tensors = []
        # 生成不同大小的复数张量（一维）
        for i in range(12, 24):  # 调整范围以避免内存溢出
            size = 2 ** i
            input_tensor = torch.randn(size, dtype=torch.complex64)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()

    def call_op(self, input_tensor):
        # 调用fftn并传递存储的参数
        return fftn(input_tensor, s=self.s, dim=self.dim, norm=self.norm)

    def get_gbps(self, input_tensor, runtime):
        # 计算输入输出总数据量（考虑复数类型）
        input_element_size = input_tensor.element_size()
        num_elements = input_tensor.numel()
        
        # 根据输入类型确定输出类型大小
        if input_tensor.is_complex():
            output_element_size = input_element_size  # 复数输出
        else:
            output_element_size = torch.tensor([], dtype=torch.complex64).element_size()  # 实转复输出
        
        total_bytes = num_elements * (input_element_size + output_element_size)
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tensor, runtime):
        # 基于FFT的5NlogN浮点运算估算
        N = input_tensor.numel()
        if N == 0:
            return 0.0
        log2_N = math.log2(N)
        flops = 5 * N * log2_N
        TFLOPS = flops / (runtime / 1000) / 1e12
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
