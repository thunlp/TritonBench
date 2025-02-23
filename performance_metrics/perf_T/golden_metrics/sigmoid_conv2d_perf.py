import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.sigmoid_conv2d import sigmoid_conv2d
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('sigmoid_conv2d', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        batch_size = 1
        in_channels = 64
        out_channels = 128
        kernel_size = 3
        # 生成不同输入尺寸：32x32, 64x64, 128x128, 256x256
        for i in range(4, 11):
            hw = 2 ** i
            input_tensor = torch.randn(batch_size, in_channels, hw, hw, dtype=self.dtype)
            weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=self.dtype)
            bias = torch.randn(out_channels, dtype=self.dtype)
            self.input_tensors.append((input_tensor, weight, bias))
    
    def to_cuda(self, input_tuple):
        input_tensor, weight, bias = input_tuple
        return (input_tensor.cuda(), weight.cuda(), bias.cuda())
    
    def call_op(self, input_tuple):
        input_tensor, weight, bias = input_tuple
        # 固定参数：stride=1, padding=0, dilation=1, groups=1
        return sigmoid_conv2d(input_tensor, weight, bias, stride=1, padding=0)
    
    def get_gbps(self, input_tuple, runtime):
        input_tensor, weight, bias = input_tuple
        element_size = input_tensor.element_size()
        N, C, H_in, W_in = input_tensor.shape
        O, _, K, _ = weight.shape
        
        # 计算输出尺寸
        H_out = H_in - K + 1
        W_out = W_in - K + 1
        
        # 计算数据量（单位：字节）
        input_bytes = input_tensor.numel() * element_size
        weight_bytes = weight.numel() * element_size
        bias_bytes = bias.numel() * element_size if bias is not None else 0
        output_bytes = N * O * H_out * W_out * element_size
        
        total_bytes = input_bytes + weight_bytes + bias_bytes + output_bytes * 3
        GBPS = total_bytes / (runtime / 1000) / 1e9  # 转换为GB/s
        return GBPS
    
    def get_tflops(self, input_tuple, runtime):
        input_tensor, weight, bias = input_tuple
        N, C, H_in, W_in = input_tensor.shape
        O, _, K, _ = weight.shape
        
        # 计算输出尺寸
        H_out = H_in - K + 1
        W_out = W_in - K + 1
        
        # 卷积FLOPs（乘加算2次操作）
        conv_flops = 2 * N * O * H_out * W_out * C * K * K
        # 偏置FLOPs
        bias_flops = N * O * H_out * W_out if bias is not None else 0
        # Sigmoid FLOPs（假设每个元素3次操作）
        sigmoid_flops = 3 * N * O * H_out * W_out
        
        total_flops = conv_flops + bias_flops + sigmoid_flops
        TFLOPS = total_flops / (runtime / 1000) / 1e12  # 转换为TFLOPS
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
