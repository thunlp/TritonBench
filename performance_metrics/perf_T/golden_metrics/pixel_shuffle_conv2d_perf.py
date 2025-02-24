import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.pixel_shuffle_conv2d import pixel_shuffle_conv2d
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('pixel_shuffle_conv2d', dtype=dtype, is_backward=is_backward, **kwargs)
        self.upscale_factor = 2  # 固定上采样因子
        self.kernel_size = 3     # 固定卷积核尺寸
        self.in_channels = 3     # 固定输入通道数

    def get_input_tensors(self):
        """生成不同尺寸的四维输入张量（NCHW格式）及对应的权重和偏置"""
        self.input_tensors = []
        # 生成空间尺寸从64x64到1024x1024的输入
        for spatial_exp in range(4, 11):  # 2^6=64, 2^10=1024
            H = W = 2 ** spatial_exp
            # 输入张量 (batch_size=1, in_channels=3, H, W)
            input_tensor = torch.randn(16, self.in_channels, H, W, dtype=torch.float32)
            
            # 权重张量 (out_channels=3*4=12, in_channels=3, kernel_size=3x3)
            out_channels = self.in_channels * (self.upscale_factor ** 2)
            weight = torch.randn(out_channels, self.in_channels, self.kernel_size, self.kernel_size, 
                                dtype=torch.float32)
            
            # 偏置张量
            bias = torch.randn(out_channels, dtype=torch.float32)
            
            self.input_tensors.append((input_tensor, weight, bias))

    def to_cuda(self, input_tuple):
        """将CPU张量元组迁移到CUDA"""
        input_tensor, weight, bias = input_tuple
        return (input_tensor.cuda(), weight.cuda(), bias.cuda())
    
    def call_op(self, input_tuple):
        """执行带参数的卷积像素重排操作"""
        input_tensor, weight, bias = input_tuple
        return pixel_shuffle_conv2d(
            input_tensor, 
            weight, 
            bias,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            upscale_factor=self.upscale_factor
        )
    
    def get_gbps(self, input_tuple, runtime):
        """计算内存带宽（GB/s）"""
        input_tensor, weight, _ = input_tuple
        N, C_in, H_in, W_in = input_tensor.shape
        
        # 计算卷积后尺寸
        H_conv = H_in - self.kernel_size + 1
        W_conv = W_in - self.kernel_size + 1
        
        # 计算像素重排后尺寸
        H_out = H_conv * self.upscale_factor
        W_out = W_conv * self.upscale_factor
        C_out = weight.shape[0] // (self.upscale_factor ** 2)
        
        # 计算数据总量（输入+输出）
        input_bytes = input_tensor.numel() * input_tensor.element_size()
        output_bytes = N * C_out * H_out * W_out * input_tensor.element_size()
        total_bytes = input_bytes * 3 + output_bytes
        
        # 计算带宽（转换为秒需要除以1000）
        return total_bytes / (runtime / 1000) / 1e9

    def get_tflops(self, input_tuple, runtime):
        """计算计算吞吐量（TFLOPS）"""
        input_tensor, weight, _ = input_tuple
        N, C_in, H_in, W_in = input_tensor.shape
        
        # 卷积输出尺寸
        H_conv = H_in - self.kernel_size + 1
        W_conv = W_in - self.kernel_size + 1
        
        # 卷积运算量计算（乘加算两次操作）
        flops_per_output = C_in * (self.kernel_size ** 2) * 2  # 每个输出点的计算量
        total_flops = N * weight.shape[0] * H_conv * W_conv * flops_per_output
        
        # 转换为TFLOPS（转换为秒需要除以1000）
        return total_flops / (runtime / 1000) / 1e12


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
