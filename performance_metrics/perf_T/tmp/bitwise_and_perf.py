import sys
import os
import json

sys.path.append('/home/lishangzhan/triton/torch_performance/GPU_efficiency/output_DeepSeek-R1')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bitwise_and import bitwise_and
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('bitwise_and', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):  # 从4KB到256MB
            size = 2 ** i
            # 生成两个相同大小的整型张量（int32）
            input_tensor = torch.randint(0, 2, (size,), dtype=torch.int32)
            other_tensor = torch.randint(0, 2, (size,), dtype=torch.int32)
            self.input_tensors.append((input_tensor, other_tensor))

    def to_cuda(self, input_tensor):
        # 将元组中的每个张量转移到CUDA
        return (input_tensor[0].cuda(), input_tensor[1].cuda())
    
    def call_op(self, input_tensor):
        # 调用二元算子，传入两个CUDA张量
        return bitwise_and(input_tensor[0], input_tensor[1])
    
    def get_gbps(self, input_tensor, runtime):
        # 计算总内存带宽: 输入*2 + 输出*1 = 3倍数据量
        numel = input_tensor[0].numel()
        element_size = input_tensor[0].element_size()
        total_bytes = 3 * numel * element_size  # 3 * num_elements * bytes_per_element
        GBPS = total_bytes / (runtime / 1000) / 1e9  # 转换为GB/s
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        # 每个元素执行一次位运算，视为1次操作
        numel = input_tensor[0].numel()
        TFLOPS = numel / (runtime / 1000) / 1e12  # 转换为Tera OPS/s
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
