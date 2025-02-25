import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.fused_instance_norm_selu_conv2d import fused_instance_norm_selu_conv2d
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_instance_norm_selu_conv2d', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        sizes = []
        for size in range(5, 11):
            sizes.append((2 ** (size - 5), 2 ** (size - 2), 2 ** size, 2 ** size))
        for N, C_in, H, W in sizes:
            input_tensor = torch.randn(N, C_in, H, W, dtype=torch.float32)
            C_out = 64
            kernel_size = 3
            weight = torch.randn(C_out, C_in, kernel_size, kernel_size, dtype=torch.float32)
            bias = torch.randn(C_out, dtype=torch.float32)
            self.input_tensors.append((input_tensor, weight, bias))

    def to_cuda(self, input_tuple):
        input_tensor, weight, bias = input_tuple
        return (input_tensor.cuda(), weight.cuda(), bias.cuda())

    def call_op(self, input_tuple):
        input_tensor, weight, bias = input_tuple
        return fused_instance_norm_selu_conv2d(
            input_tensor, weight, bias, 
            stride=1, padding=0, dilation=1, groups=1,
            eps=1e-05, momentum=0.1, affine=False, track_running_stats=False
        )

    def get_gbps(self, input_tuple, runtime):
        input_tensor, weight, bias = input_tuple
        input_bytes = input_tensor.numel() * input_tensor.element_size()
        weight_bytes = weight.numel() * weight.element_size()
        bias_bytes = bias.numel() * bias.element_size() if bias is not None else 0
        
        _, _, H, W = input_tensor.shape
        K = weight.shape[2]  # kernel_size
        padding = 0
        stride = 1
        dilation = 1
        H_out = (H + 2*padding - dilation*(K-1) - 1) // stride + 1
        W_out = (W + 2*padding - dilation*(K-1) - 1) // stride + 1
        output_size = input_tensor.shape[0] * weight.shape[0] * H_out * W_out
        output_bytes = output_size * input_tensor.element_size()
        
        total_bytes = input_bytes + weight_bytes + bias_bytes + output_bytes * 5
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tuple, runtime):
        input_tensor, weight, bias = input_tuple
        N, C_in, H, W = input_tensor.shape
        C_out, _, K, _ = weight.shape
        
        padding = 0
        stride = 1
        dilation = 1
        H_out = (H + 2*padding - dilation*(K-1) - 1) // stride + 1
        W_out = (W + 2*padding - dilation*(K-1) - 1) // stride + 1
        
        conv_flops = 2 * C_in * K * K * N * C_out * H_out * W_out
        if bias is not None:
            conv_flops += N * C_out * H_out * W_out
        
        selu_flops = N * C_out * H_out * W_out
        
        instance_norm_flops = 6 * N * C_out * H_out * W_out
        
        total_flops = conv_flops + selu_flops + instance_norm_flops
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
