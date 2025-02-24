import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.relu_max_pool2d_conv2d import relu_max_pool2d_conv2d
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('relu_max_pool2d_conv2d', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        in_channels = 3
        out_channels = 64
        kernel_size = 3
        # 生成不同尺寸的输入，保证卷积和池化后的尺寸为整数
        for s in range(5, 14):  # s为池化后的尺寸，对应输入H=2*s+2
            H = W = 2 ** s + 2  # 保证H和W相同
            input_tensor = torch.randn(1, in_channels, H, W, dtype=self.dtype or torch.float32)
            weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=self.dtype or torch.float32)
            bias = torch.randn(out_channels, dtype=self.dtype or torch.float32)
            self.input_tensors.append((input_tensor, weight, bias))

    def to_cuda(self, input_tuple):
        input_tensor, weight, bias = input_tuple
        return (input_tensor.cuda(), weight.cuda(), bias.cuda())

    def call_op(self, input_tuple):
        input_tensor, weight, bias = input_tuple
        return relu_max_pool2d_conv2d(
            input_tensor, 
            weight, 
            bias,
            conv_stride=1,
            conv_padding=0,
            pool_kernel_size=2,
            pool_stride=2
        )

    def get_gbps(self, input_tuple, runtime):
        input_tensor, weight, bias = input_tuple
        element_size = input_tensor.element_size()
        H = input_tensor.size(2)
        W = input_tensor.size(3)
        s = (H - 2) // 2  # 池化后的尺寸
        
        # 计算各部分的字节数
        input_bytes = input_tensor.numel() * element_size
        weight_bytes = weight.numel() * element_size
        bias_bytes = bias.numel() * element_size if bias is not None else 0
        output_numel = 64 * s * s  # 输出通道数为64
        output_bytes = output_numel * element_size
        
        total_bytes = input_bytes * 3 + weight_bytes + bias_bytes + output_bytes * 3
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tuple, runtime):
        input_tensor, weight, bias = input_tuple
        H = input_tensor.size(2)
        W = input_tensor.size(3)
        conv_output_H = H - 2
        conv_output_W = W - 2
        
        # 卷积部分FLOPs
        conv_output_elements = 64 * conv_output_H * conv_output_W
        conv_flops = 2 * 3 * 3 * 3 * conv_output_elements + conv_output_elements  # 乘加操作+bias
        
        # 池化部分FLOPs
        s = conv_output_H // 2
        pool_output_elements = 64 * s * s
        pool_flops = 3 * pool_output_elements  # 每个窗口3次比较
        
        # ReLU部分FLOPs
        relu_flops = pool_output_elements
        
        total_flops = conv_flops + pool_flops + relu_flops
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
