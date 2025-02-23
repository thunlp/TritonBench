import sys
import os
import json

sys.path.append('/home/lishangzhan/triton/torch_performance/GPU_efficiency/output_DeepSeek-R1')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scaled_add_norm import scaled_add_norm
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('scaled_add_norm', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        # 生成不同大小的输入张量对(y, x)及alpha
        for i in range(12, 28):
            size = 2 ** i
            # 根据dtype参数生成对应类型的张量
            dtype = self.dtype or torch.float32
            y = torch.rand(size, dtype=dtype)
            x = torch.rand(size, dtype=dtype)
            alpha = 0.5  # 固定alpha值，也可随机生成
            self.input_tensors.append((y, x, alpha))

    def to_cuda(self, input_tensor):
        # 将输入张量转移到CUDA设备
        y, x, alpha = input_tensor
        return (y.cuda(), x.cuda(), alpha)
    
    def call_op(self, input_tensor):
        # 调用算子并返回结果
        y, x, alpha = input_tensor
        return scaled_add_norm(y, x, alpha)
    
    def get_gbps(self, input_tensor, runtime):
        # 计算内存带宽（GBPS）
        y, x, alpha = input_tensor
        total_bytes = 5 * y.numel() * y.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9  # 转换为GB/s
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        # 计算浮点性能（TFLOPS）
        y, x, alpha = input_tensor
        # 每个元素在scaled_add中需2次浮点操作（乘加）
        # 在norm中需2次浮点操作（平方和累加）
        flops_per_element = 2 + 2
        total_flops = flops_per_element * y.numel()
        TFLOPS = total_flops / (runtime / 1000) / 1e12  # 转换为TFLOPS
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
