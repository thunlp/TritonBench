import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.cos import cos  # 正确引入cos算子
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        # 修正算子名称为'cos'（原模板中的'cos.py'不正确）
        super().__init__('cos', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        """生成不同规模的CPU张量，范围2^12到2^27"""
        self.input_tensors = []
        for i in range(12, 28):
            size = 2 ** i
            input_tensor = torch.rand(size, dtype=torch.float32)  # 使用float32类型
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        """将张量转移到CUDA设备"""
        return input_tensor.cuda()
    
    def call_op(self, input_tensor):
        """调用cos算子进行计算"""
        return cos(input_tensor)
    
    def get_gbps(self, input_tensor, runtime):
        """计算内存带宽（GB/s）"""
        # 总数据量 = 输入张量大小 + 输出张量大小（各为numel * element_size）
        total_bytes = input_tensor.numel() * input_tensor.element_size() * 2
        # 转换runtime单位（毫秒->秒）并计算GB/s
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        """计算计算吞吐量（TFLOPS）"""
        # 假设每个元素需要一次浮点运算（根据三角函数计算复杂度可能调整）
        FLOPS = input_tensor.numel()
        # 转换runtime单位（毫秒->秒）并计算TFLOPS
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
