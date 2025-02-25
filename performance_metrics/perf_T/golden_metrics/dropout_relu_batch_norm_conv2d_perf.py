import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.dropout_relu_batch_norm_conv2d import dropout_relu_batch_norm_conv2d
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('dropout_relu_batch_norm_conv2d', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        C_in = 3
        C_out = 64
        kH, kW = 3, 3
        groups = 1
        
        for exp in range(5, 12):
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
        return dropout_relu_batch_norm_conv2d(
            input_tensor, weight, bias,
            stride=1, padding=0, dilation=1,
            groups=1, p=0.5, training=True, inplace=False
        )
    
    def get_gbps(self, input_tuple, runtime):
        input_tensor, weight, bias = input_tuple
        input_bytes = input_tensor.numel() * input_tensor.element_size()
        weight_bytes = weight.numel() * weight.element_size()
        bias_bytes = bias.numel() * bias.element_size() if bias is not None else 0
        
        N, C_in, H_in, W_in = input_tensor.shape
        C_out = weight.shape[0]
        kH, kW = weight.shape[2], weight.shape[3]
        H_out = (H_in - kH) // 1 + 1  # stride=1, padding=0
        W_out = (W_in - kW) // 1 + 1
        output_bytes = N * C_out * H_out * W_out * input_tensor.element_size()
        
        total_bytes = (input_bytes + weight_bytes + bias_bytes + output_bytes) * 4
        return total_bytes / (runtime / 1000) / 1e9  # GBPS
    
    def get_tflops(self, input_tuple, runtime):
        input_tensor, weight, _ = input_tuple
        N, C_in, H_in, W_in = input_tensor.shape
        C_out = weight.shape[0]
        kH, kW = weight.shape[2], weight.shape[3]
        groups = 1
        
        H_out = (H_in - kH) // 1 + 1
        W_out = (W_in - kW) // 1 + 1
        
        conv_flops = 2 * C_in * kH * kW * C_out * H_out * W_out // groups
        
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
