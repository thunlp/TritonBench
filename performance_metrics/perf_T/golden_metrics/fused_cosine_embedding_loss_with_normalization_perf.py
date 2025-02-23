import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.fused_cosine_embedding_loss_with_normalization import fused_cosine_embedding_loss_with_normalization
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_cosine_embedding_loss_with_normalization', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        # 生成不同batch size的测试数据（CPU Tensor）
        for i in range(12, 21):  # 控制规模防止OOM
            N = 2 ** i          # batch size
            D = 256             # 特征维度
            input1 = torch.randn(N, D, dtype=torch.float32)
            input2 = torch.randn(N, D, dtype=torch.float32)
            target = torch.randint(0, 2, (N,), dtype=torch.float32) * 2 - 1  # 生成±1标签
            self.input_tensors.append((input1, input2, target))

    def to_cuda(self, input_tuple):
        # 将元组中的每个Tensor转移到CUDA
        input1, input2, target = input_tuple
        return (input1.cuda(), input2.cuda(), target.cuda())
    
    def call_op(self, input_tuple):
        # 调用算子并返回结果
        input1, input2, target = input_tuple
        return fused_cosine_embedding_loss_with_normalization(input1, input2, target)

    def get_gbps(self, input_tensor, runtime):
        # 计算内存带宽（GB/s）
        input1, input2, target = input_tensor
        element_size = input1.element_size()  # 获取元素字节大小
        
        # 总数据量 = 输入数据 + 输出数据
        input_bytes = (input1.numel() * 10 + target.numel()) * element_size
        output_bytes = 1 * element_size  # reduction='mean'时输出为标量
        total_bytes = (input_bytes + output_bytes)
        
        # 计算GBPS（考虑毫秒到秒的转换）
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        # 计算计算吞吐量（TFLOPS）
        input1, input2, target = input_tensor
        N, D = input1.shape
        
        # FLOPS估算（归一化2次: 6ND，点积: 2ND，后续计算: 3N）
        total_flops = 8 * N * D  # 近似估算
        
        # 转换为TFLOPS（考虑毫秒到秒的转换）
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
