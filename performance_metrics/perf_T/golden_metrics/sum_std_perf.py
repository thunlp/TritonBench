import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.sum_std import sum_std
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('sum_std', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        # 生成不同大小的一维张量（类似abs示例）
        # 范围从2^12到2^24避免显存不足
        for i in range(12, 25):
            size = 2 ** i
            input_tensor = torch.rand(size, dtype=torch.float32)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()
    
    def call_op(self, input_tensor):
        # 调用时指定dim=None（全量求和）
        return sum_std(input_tensor, dim=None)
    
    def get_gbps(self, input_tensor, runtime):
        # 输入输出总数据量（输入+标量输出）
        input_bytes = input_tensor.numel() * input_tensor.element_size()
        output_bytes = 1 * input_tensor.element_size()  # 标量输出
        total_bytes = input_bytes + output_bytes
        
        # 转换为GB/s（注意runtime单位是毫秒）
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        # 近似计算FLOPS：
        # 1. sum操作需要numel次访存（每个元素参与计算）
        # 2. std计算中均值、方差等操作量级远小于sum，故近似忽略
        FLOPS = input_tensor.numel()  
        
        # 转换为TFLOPS（注意runtime单位是毫秒）
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
