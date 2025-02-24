import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.min_gelu import min_gelu
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, approximate='none', **kwargs):
        super().__init__('min_gelu', dtype=dtype, is_backward=is_backward, **kwargs)
        self.approximate = approximate  # 控制GELU近似方式

    def get_input_tensors(self):
        self.input_tensors = []
        # 生成不同size的1D输入张量 (2^12 到 2^28 元素)
        for i in range(12, 28):
            size = 2 ** i
            input_tensor = torch.rand(size, dtype=self.dtype)  # 使用类指定的dtype
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()  # 移动张量到CUDA设备
    
    def call_op(self, input_tensor):
        # 固定dim=0进行归约，保持测试一致性
        return min_gelu(input_tensor, dim=0, approximate=self.approximate)
    
    def get_gbps(self, input_tensor, runtime):
        """计算内存带宽 (GB/s)"""
        element_size = input_tensor.element_size()
        # 总数据量 = 输入(读) + 中间结果(写+读) + 输出(写)
        total_bytes = (3 * input_tensor.numel() + 1) * element_size  # +1为标量输出
        GBPS = total_bytes / (runtime / 1000) / 1e9  # 转换为GB/s
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        """计算计算吞吐量 (TFLOPS)"""
        # 根据approximate模式选择FLOPS系数
        if self.approximate == 'none':
            gelu_flops_per_element = 8  # erf精确计算的估计值
        else:
            gelu_flops_per_element = 12  # tanh近似的估计值
            
        num_elements = input_tensor.numel()
        # GELU计算 + MIN归约
        total_flops = num_elements * gelu_flops_per_element + (num_elements - 1)
        TFLOPS = total_flops / (runtime / 1000) / 1e12  # 转换为TFLOPS
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
