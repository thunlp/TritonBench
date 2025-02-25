import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.fused_layer_norm_relu_linear import fused_layer_norm_relu_linear
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_layer_norm_relu_linear', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(8, 18):
            in_features = 2 ** i
            out_features = 2 ** (i - 1)
            batch_size = 32
            input_tensor = torch.randn(batch_size, in_features, dtype=torch.float32)
            weight = torch.randn(out_features, in_features, dtype=torch.float32)
            bias = torch.randn(out_features, dtype=torch.float32)
            normalized_shape = (out_features,)
            self.input_tensors.append((input_tensor, weight, bias, normalized_shape))

    def to_cuda(self, input_tuple):
        input_tensor, weight, bias, normalized_shape = input_tuple
        return (
            input_tensor.cuda(),
            weight.cuda(),
            bias.cuda(),
            normalized_shape
        )
    
    def call_op(self, input_tuple):
        input_tensor, weight, bias, normalized_shape = input_tuple
        return fused_layer_norm_relu_linear(
            input_tensor, 
            weight, 
            bias, 
            normalized_shape=normalized_shape
        )
    
    def get_gbps(self, input_tuple, runtime):
        input_tensor, weight, bias, _ = input_tuple
        input_bytes = input_tensor.numel() * input_tensor.element_size()
        weight_bytes = weight.numel() * weight.element_size()
        bias_bytes = bias.numel() * bias.element_size() if bias is not None else 0
        output_bytes = input_tensor.shape[0] * weight.shape[0] * input_tensor.element_size()
        
        total_bytes = input_bytes + weight_bytes + bias_bytes + output_bytes * 5
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tuple, runtime):
        input_tensor, weight, _, _ = input_tuple
        batch_size, in_features = input_tensor.shape
        out_features = weight.shape[0]
        
        flops_linear = 2 * batch_size * out_features * in_features
        
        # ReLU FLOPs: B*Out
        flops_relu = batch_size * out_features
        
        flops_layernorm = 8 * batch_size * out_features
        
        total_flops = flops_linear + flops_relu + flops_layernorm
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
