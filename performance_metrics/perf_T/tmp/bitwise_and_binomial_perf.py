import sys
import os
import json

sys.path.append('/home/lishangzhan/triton/torch_performance/GPU_efficiency/output_DeepSeek-R1_rag')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bitwise_and_binomial import bitwise_and_binomial
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('bitwise_and_binomial', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):
            size = 2 ** i
            # 生成输入张量（int32类型）
            input_tensor = torch.randint(0, 2, (size,), dtype=torch.int32)
            other_tensor = torch.randint(0, 2, (size,), dtype=torch.int32)
            # 生成Binomial参数（固定total_count=10，随机probs）
            total_count = torch.tensor(10, dtype=torch.int32)
            probs = torch.rand(size, dtype=torch.float32) * 0.5 + 0.25  # 0.25-0.75范围
            self.input_tensors.append((input_tensor, other_tensor, total_count, probs))

    def to_cuda(self, input_tuple):
        # 将每个张量转移到CUDA
        input_tensor, other_tensor, total_count, probs = input_tuple
        return (
            input_tensor.cuda(),
            other_tensor.cuda(),
            total_count.cuda(),
            probs.cuda()
        )

    def call_op(self, input_tuple):
        # 解包参数并调用算子
        input_tensor, other_tensor, total_count, probs = input_tuple
        return bitwise_and_binomial(input_tensor, other_tensor, total_count, probs=probs)

    def get_gbps(self, input_tuple, runtime):
        # 计算总数据量（输入+输出）
        input_tensor, other_tensor, total_count, probs = input_tuple
        input_bytes = input_tensor.numel() * input_tensor.element_size()
        other_bytes = other_tensor.numel() * other_tensor.element_size()
        total_count_bytes = total_count.numel() * total_count.element_size()
        probs_bytes = probs.numel() * probs.element_size()
        output_bytes = input_tensor.numel() * 8  # 输出为int64类型
        
        total_bytes = input_bytes + other_bytes + total_count_bytes + probs_bytes + output_bytes
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tuple, runtime):
        # 计算浮点运算量（基于Binomial采样次数）
        input_tensor, other_tensor, total_count, _ = input_tuple
        numel = input_tensor.numel()
        operations_per_element = total_count.item()  # 每个元素的采样次数
        total_ops = numel * operations_per_element
        TFLOPS = total_ops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
