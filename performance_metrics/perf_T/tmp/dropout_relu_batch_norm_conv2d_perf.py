import sys
import os
import json

sys.path.append('/home/lishangzhan/triton/torch_performance/GPU_efficiency/output_DeepSeek-R1')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dropout_relu_batch_norm_conv2d import dropout_relu_batch_norm_conv2d
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('dropout_relu_batch_norm_conv2d', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        C_in = 3  # 输入通道数
        C_out = 64  # 输出通道数
        kH, kW = 3, 3  # 卷积核尺寸
        groups = 1  # 分组卷积参数
        
        # 生成不同尺寸的输入张量（基于H/W）
        for exp in range(5, 12):  # H/W从32到2048
            H = W = 2 ** exp
            input_tensor = torch.randn(1, C_in, H, W, dtype=self.dtype)
            weight = torch.randn(C_out, C_in // groups, kH, kW, dtype=self.dtype)
            bias = torch.randn(C_out, dtype=self.dtype)
            self.input_tensors.append((input_tensor, weight, bias))
    
    def to_cuda(self, input_tuple):
        input_tensor, weight, bias = input_tuple
        return (input_tensor.cuda(), weight.cuda(), bias.cuda())
        
    def call_op(self, input_tuple):
        input_tensor, weight, bias = input_tuple
        # 使用固定参数调用组合算子
        return dropout_relu_batch_norm_conv2d(
            input_tensor, weight, bias,
            stride=1, padding=0, dilation=1,
            groups=1, p=0.5, training=True, inplace=False
        )
    
    def get_gbps(self, input_tuple, runtime):
        input_tensor, weight, bias = input_tuple
        # 计算输入张量+权重+偏置的字节数
        input_bytes = input_tensor.numel() * input_tensor.element_size()
        weight_bytes = weight.numel() * weight.element_size()
        bias_bytes = bias.numel() * bias.element_size() if bias is not None else 0
        
        # 计算输出张量尺寸
        N, C_in, H_in, W_in = input_tensor.shape
        C_out = weight.shape[0]
        kH, kW = weight.shape[2], weight.shape[3]
        H_out = (H_in - kH) // 1 + 1  # stride=1, padding=0
        W_out = (W_in - kW) // 1 + 1
        output_bytes = N * C_out * H_out * W_out * input_tensor.element_size()
        
        # 总数据量（输入+权重+偏置+输出）
        total_bytes = (input_bytes + weight_bytes + bias_bytes + output_bytes) * 4
        return total_bytes / (runtime / 1000) / 1e9  # GBPS
    
    def get_tflops(self, input_tuple, runtime):
        input_tensor, weight, _ = input_tuple
        N, C_in, H_in, W_in = input_tensor.shape
        C_out = weight.shape[0]
        kH, kW = weight.shape[2], weight.shape[3]
        groups = 1
        
        # 计算输出特征图尺寸
        H_out = (H_in - kH) // 1 + 1
        W_out = (W_in - kW) // 1 + 1
        
        # 卷积层FLOPs（乘加算两次操作）
        conv_flops = 2 * C_in * kH * kW * C_out * H_out * W_out // groups
        
        # 其他层FLOPs（BN约4次/元素，ReLU 1次，Dropout 2次）
        output_elements = N * C_out * H_out * W_out
        bn_flops = 4 * output_elements
        relu_flops = 1 * output_elements
        dropout_flops = 2 * output_elements
        
        total_flops = conv_flops + bn_flops + relu_flops + dropout_flops
        return total_flops / (runtime / 1000) / 1e12  # TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
