import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.fused_gather_masked_fill import fused_gather_masked_fill
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_gather_masked_fill', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        # 生成不同尺寸的测试用例（调整范围防止OOM）
        for i in range(5, 13):  # 2^12到2^20
            S = 2 ** i
            # 输入张量（CPU）
            input_tensor = torch.randn((S, 1024), dtype=torch.float32)
            # 索引张量（确保在有效范围内）
            index = torch.randint(0, S, (S, 1024), dtype=torch.int64)
            # 布尔掩码（与输出同形）
            mask = torch.rand((S, 1024)) > 0.5
            value = 0.5
            dim = 0
            self.input_tensors.append((input_tensor, dim, index, mask, value))

    def to_cuda(self, input_tuple):
        # 转移各张量到CUDA
        input_tensor, dim, index, mask, value = input_tuple
        return (
            input_tensor.cuda(),
            dim,
            index.cuda(),
            mask.cuda(),
            value
        )
    
    def call_op(self, input_tuple):
        # 解包参数调用算子
        return fused_gather_masked_fill(*input_tuple)
    
    def get_gbps(self, input_tuple, runtime):
        # 计算总内存访问量
        input_tensor, _, index, mask, _ = input_tuple
        
        input_bytes = input_tensor.numel() * input_tensor.element_size()
        index_bytes = index.numel() * index.element_size()
        mask_bytes = mask.numel() * mask.element_size()
        output_bytes = index.numel() * input_tensor.element_size()  # 输出与index同形
        
        total_bytes = input_bytes + index_bytes + mask_bytes + output_bytes
        return total_bytes / (runtime / 1000) / 1e9  # 转换为GB/s

    def get_tflops(self, input_tuple, runtime):
        # 以输出元素数作为操作数
        _, _, index, _, _ = input_tuple
        flops = index.numel()  # 每个输出元素对应一次操作
        return flops / (runtime / 1000) / 1e12  # 转换为TFLOPS
    
if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
