import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.relu_batch_norm_conv2d import relu_batch_norm_conv2d
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('relu_batch_norm_conv2d', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        batch_size = 1
        base_channels = 64
        kernel_size = 3
        padding = 1
        
        for i in range(5):
            H = W = 32 * (2 ** i)
            in_channels = base_channels * (2 ** i)
            out_channels = base_channels * (2 ** (i+1))
            
            input_tensor = torch.randn(batch_size, in_channels, H, W, dtype=torch.float32)
            
            weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.float32)
            
            running_mean = torch.zeros(out_channels)
            running_var = torch.ones(out_channels)
            bn_weight = torch.ones(out_channels)
            bn_bias = torch.zeros(out_channels)
            
            params = (
                input_tensor,
                weight,
                None,        # bias
                1,           # stride
                padding,     # padding
                1,           # dilation
                1,           # groups
                running_mean,
                running_var,
                bn_weight,
                bn_bias,
                False,       # training
                0.1,         # momentum
                1e-5,        # eps
                False        # inplace
            )
            self.input_tensors.append(params)
    
    def to_cuda(self, input_tuple):
        cuda_params = []
        for param in input_tuple:
            if isinstance(param, torch.Tensor):
                cuda_params.append(param.cuda())
            else:
                cuda_params.append(param)
        return tuple(cuda_params)
    
    def call_op(self, input_tuple):
        return relu_batch_norm_conv2d(*input_tuple)
    
    def get_gbps(self, input_tuple, runtime):
        input_tensor = input_tuple[0]
        weight = input_tuple[1]
        
        input_bytes = input_tensor.numel() * input_tensor.element_size()
        output = self.call_op(input_tuple)
        output_bytes = output.numel() * output.element_size()
        
        total_bytes = input_bytes + output_bytes
        
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tuple, runtime):
        input_tensor = input_tuple[0]
        weight = input_tuple[1]
        N, C_in, H, W = input_tensor.shape
        C_out = weight.shape[0]
        K = weight.shape[2]
        
        # 1. Conv2d: 2*C_in*K*K*C_out*H*W*N
        conv_flops = 2 * C_in * K**2 * C_out * H * W * N
        
        # 2. BatchNorm: 5*C_out*H*W*N
        bn_flops = 5 * C_out * H * W * N
        
        # 3. ReLU: 1*C_out*H*W*N
        relu_flops = C_out * H * W * N
        
        total_flops = conv_flops + bn_flops + relu_flops
        
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
