import sys
import os
import json

sys.path.append('/home/lishangzhan/triton/torch_performance/GPU_efficiency/output_DeepSeek-R1')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from log import log
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('log', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):
            size = 2 ** i
            # 生成正数输入避免log(0)，使用float32类型
            input_tensor = torch.rand(size, dtype=torch.float32) + 1e-6
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()
        
    def call_op(self, input_tensor):
        # 使用预分配输出张量来避免内存分配开销
        out = torch.empty_like(input_tensor)
        return log(input_tensor, out=out)
    
    def get_gbps(self, input_tensor, runtime):
        # 计算总数据量（输入+输出）
        total_bytes = input_tensor.numel() * input_tensor.element_size() * 2
        # 转换为GB/s（runtime单位是毫秒）
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        # 假设每个元素需要20次浮点运算（典型超越函数估算值）
        # 可根据实际情况调整该系数
        FLOPS_PER_ELEMENT = 20
        total_flops = input_tensor.numel() * FLOPS_PER_ELEMENT
        # 转换为TFLOPS
        TFLOPS = total_flops / (runtime / 1000) / 1e12
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
