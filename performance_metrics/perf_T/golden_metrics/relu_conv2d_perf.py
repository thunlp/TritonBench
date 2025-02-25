import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.relu_conv2d import relu_conv2d
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('relu_conv2d', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(5, 12):
            H = W = 2 ** i
            input_tensor = torch.randn(1, 3, H, W, dtype=torch.float32)
            weight_tensor = torch.randn(64, 3, 3, 3, dtype=torch.float32)
            self.input_tensors.append((input_tensor, weight_tensor))

    def to_cuda(self, input_tuple):
        input_tensor, weight_tensor = input_tuple
        return (input_tensor.cuda(), weight_tensor.cuda())
    
    def call_op(self, input_tuple):
        input_tensor, weight_tensor = input_tuple
        return relu_conv2d(input_tensor, weight_tensor, 
                          stride=1, padding=1, inplace=False)

    def get_gbps(self, input_tuple, runtime):
        input_tensor, weight_tensor = input_tuple
        element_size = input_tensor.element_size()  # 4 bytes for float32
        
        input_bytes = input_tensor.numel() * element_size
        weight_bytes = weight_tensor.numel() * element_size
        
        output_shape = (input_tensor.size(0), weight_tensor.size(0), 
                       input_tensor.size(2), input_tensor.size(3))
        output_bytes = torch.zeros(output_shape).numel() * element_size
        
        total_bytes = input_bytes + weight_bytes + output_bytes * 3
        
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tuple, runtime):
        input_tensor, weight_tensor = input_tuple
        N, C_in, H, W = input_tensor.shape
        C_out, _, K, _ = weight_tensor.shape  # K=3
        
        H_out, W_out = H, W
        
        conv_flops = N * C_out * H_out * W_out * C_in * K * K * 2
        
        relu_flops = N * C_out * H_out * W_out
        
        total_flops = conv_flops + relu_flops
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
