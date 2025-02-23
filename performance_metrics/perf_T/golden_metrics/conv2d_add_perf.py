import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.conv2d_add import conv2d_add
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('conv2d_add', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        # 生成不同尺寸的测试用例（从32x32到1024x1024）
        for i in range(5, 11):  # H = 2^5=32到2^10=1024
            H = W = 2 ** i
            batch_size = 16
            in_channels = 32
            out_channels = 32
            kH = kW = 3
            stride = 1
            padding = 1
            groups = 1
            
            # 创建输入张量
            input_tensor = torch.randn(batch_size, in_channels, H, W, dtype=torch.float32)
            weight = torch.randn(out_channels, in_channels//groups, kH, kW, dtype=torch.float32)
            
            # 计算输出形状并生成other张量
            H_out = (H + 2*padding - (kH - 1)) // stride
            W_out = (W + 2*padding - (kW - 1)) // stride
            other = torch.randn(batch_size, out_channels, H_out, W_out, dtype=torch.float32)
            
            self.input_tensors.append((input_tensor, weight, other))

    def to_cuda(self, input_tuple):
        input_tensor, weight, other = input_tuple
        return (input_tensor.cuda(), weight.cuda(), other.cuda())
        
    def call_op(self, input_tuple):
        input_tensor, weight, other = input_tuple
        # 使用与输入生成一致的卷积参数
        return conv2d_add(input_tensor, weight, other=other, 
                         stride=1, padding=1, groups=1)

    def get_gbps(self, input_tuple, runtime):
        input_tensor, weight, other = input_tuple
        element_size = input_tensor.element_size()
        
        # 计算总数据量（输入+输出）
        input_bytes = input_tensor.numel() * element_size
        weight_bytes = weight.numel() * element_size
        other_bytes = other.numel() * element_size
        output_bytes = other.numel() * element_size  # output shape == other.shape
        
        total_bytes = (input_bytes + weight_bytes + other_bytes + output_bytes) * 2
        return total_bytes / (runtime / 1000) / 1e9  # GBPS

    def get_tflops(self, input_tuple, runtime):
        input_tensor, weight, other = input_tuple
        batch, in_channels, H, W = input_tensor.shape
        out_channels, in_ch_per_group, kH, kW = weight.shape
        
        # 卷积部分计算量
        output_elements = other.numel()
        conv_flops_per_element = 2 * kH * kW * in_ch_per_group  # 乘加算2次操作
        conv_flops = output_elements * conv_flops_per_element
        
        # 加法部分计算量（alpha乘法和加法）
        add_flops = output_elements * 2
        
        total_flops = conv_flops + add_flops
        return total_flops / (runtime / 1000) / 1e12  # TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
