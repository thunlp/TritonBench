import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.conv2d import conv2d
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('conv2d', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        # 不同规模测试用例：输入形状、输出通道数、卷积核尺寸
        # configs = [
        #     ((1, 3, 32, 32), 64, (3, 3)),
        #     ((4, 64, 16, 16), 128, (3, 3)),
        #     ((16, 128, 8, 8), 256, (3, 3)),
        #     ((32, 256, 8, 8), 512, (3, 3)),
        # ]
        configs = []
        for i in range(5, 12):  # H = 2^5=32到2^10=1024
            input_shape = (2 ** (i - 4), 2 ** (i - 1), 16, 16)
            out_channels = 2 ** i
            kernel_size = (3, 3)
            configs.append((input_shape, out_channels, kernel_size))
        for input_shape, out_channels, kernel_size in configs:
            # 生成CPU张量
            input_tensor = torch.randn(*input_shape, dtype=self.dtype or torch.float32)
            in_channels = input_shape[1]
            kH, kW = kernel_size
            weight = torch.randn(out_channels, in_channels, kH, kW, dtype=self.dtype or torch.float32)
            bias = torch.randn(out_channels, dtype=self.dtype or torch.float32)
            self.input_tensors.append((input_tensor, weight, bias))
    
    def to_cuda(self, input_tuple):
        # 将元组内所有张量转移到CUDA
        input_tensor, weight, bias = input_tuple
        return (input_tensor.cuda(), weight.cuda(), bias.cuda())
    
    def call_op(self, input_tuple):
        # 解包参数并调用卷积算子
        input, weight, bias = input_tuple
        return conv2d(input, weight, bias, 
                     stride=1, padding=0, dilation=1, groups=1)
    
    def get_gbps(self, input_tuple, runtime):
        # 计算总内存带宽
        input_tensor, weight, bias = input_tuple
        N, C_in, H_in, W_in = input_tensor.shape
        C_out, _, kH, kW = weight.shape
        
        # 计算输出尺寸
        H_out = H_in - kH + 1
        W_out = W_in - kW + 1
        output_numel = N * C_out * H_out * W_out
        
        # 计算总数据量
        element_size = input_tensor.element_size()
        input_bytes = input_tensor.numel() * element_size
        weight_bytes = weight.numel() * element_size
        bias_bytes = bias.numel() * element_size
        output_bytes = output_numel * element_size
        total_bytes = input_bytes + weight_bytes + bias_bytes + output_bytes
        
        # 计算GBPS（注意runtime单位是毫秒）
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tuple, runtime):
        # 计算浮点运算量
        input_tensor, weight, bias = input_tuple
        N, C_in, H_in, W_in = input_tensor.shape
        C_out, _, kH, kW = weight.shape
        groups = 1
        
        # 计算输出尺寸
        H_out = H_in - kH + 1
        W_out = W_in - kW + 1
        
        # 计算FLOPs（乘加算两次）
        flops_per_output = 2 * kH * kW * C_in // groups
        total_flops = N * C_out * H_out * W_out * flops_per_output
        
        # 加上偏置运算
        if bias is not None:
            total_flops += N * C_out * H_out * W_out  # 每个输出元素加一次
        
        # 计算TFLOPS
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
