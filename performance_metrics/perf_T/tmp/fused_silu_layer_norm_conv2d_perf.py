import sys
import os
import json

sys.path.append('/home/lishangzhan/triton/torch_performance/GPU_efficiency/output_DeepSeek-R1')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fused_silu_layer_norm_conv2d import fused_silu_layer_norm_conv2d
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_silu_layer_norm_conv2d', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        out_channels = 64  # 固定输出通道数
        kernel_size = 3
        
        for i in range(5, 10):
            batch_size, in_channels, H, W = 2 ** (i - 5), 2 ** (i - 2), 2 ** i, 2 ** i
            # 生成输入张量x
            x = torch.randn(batch_size, in_channels, H, W, dtype=torch.float32)
            # 生成卷积权重conv_weight
            conv_weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.float32)
            # 生成LayerNorm的权重
            ln_weight = torch.randn(out_channels, dtype=torch.float32)
            # 生成卷积偏置conv_bias
            conv_bias = torch.randn(out_channels, dtype=torch.float32)
            # 将参数打包为元组
            input_case = (x, ln_weight, conv_weight, conv_bias)
            self.input_tensors.append(input_case)

    def to_cuda(self, input_tuple):
        # 将每个张量转移到CUDA
        return tuple(t.cuda() for t in input_tuple)
    
    def call_op(self, input_tuple):
        # 解包参数并调用算子
        x, weight, conv_weight, conv_bias = input_tuple
        return fused_silu_layer_norm_conv2d(x, weight, conv_weight, conv_bias)

    def get_gbps(self, input_tuple, runtime):
        # 计算总数据量（输入+输出）的GBPS
        x, weight, conv_weight, conv_bias = input_tuple
        # 计算卷积输出形状
        kernel_size = conv_weight.shape[2]
        H_out = x.shape[2] - kernel_size + 1
        W_out = x.shape[3] - kernel_size + 1
        output_numel = x.shape[0] * conv_weight.shape[0] * H_out * W_out
        
        # 输入数据量
        input_bytes = x.numel() * x.element_size()
        input_bytes += weight.numel() * weight.element_size()
        input_bytes += conv_weight.numel() * conv_weight.element_size()
        if conv_bias is not None:
            input_bytes += conv_bias.numel() * conv_bias.element_size()
        # 输出数据量
        output_bytes = output_numel * x.element_size()
        total_bytes = input_bytes + output_bytes * 5
        
        # 计算GBPS
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tuple, runtime):
        # 计算总浮点运算量（TFLOPS）
        x, weight, conv_weight, conv_bias = input_tuple
        batch_size, in_channels, H_in, W_in = x.shape
        out_channels = conv_weight.shape[0]
        kernel_size = conv_weight.shape[2]
        groups = 1  # 假设groups=1
        
        # 计算输出尺寸
        H_out = H_in - kernel_size + 1
        W_out = W_in - kernel_size + 1
        
        # 卷积的FLOPs（乘加算两次）
        conv_flops = 2 * batch_size * out_channels * H_out * W_out * in_channels * kernel_size * kernel_size
        
        # LayerNorm的FLOPs（假设每个元素3次操作）
        ln_flops = 3 * batch_size * out_channels * H_out * W_out
        
        # SiLU的FLOPs（假设每个元素3次操作）
        silu_flops = 3 * batch_size * out_channels * H_out * W_out
        
        total_flops = conv_flops + ln_flops + silu_flops
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
