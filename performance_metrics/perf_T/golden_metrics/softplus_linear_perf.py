import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.softplus_linear import softplus_linear
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('softplus_linear', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 16):
            in_features = 2 ** i
            out_features = 2 ** i  # 保持输入输出维度相同
            input_tensor = torch.randn(1, in_features, dtype=self.dtype)  # batch_size=1
            weight = torch.randn(out_features, in_features, dtype=self.dtype)
            bias = torch.randn(out_features, dtype=self.dtype)
            self.input_tensors.append((input_tensor, weight, bias))

    def to_cuda(self, input_tuple):
        input_tensor, weight, bias = input_tuple
        input_cuda = input_tensor.cuda()
        weight_cuda = weight.cuda()
        bias_cuda = bias.cuda() if bias is not None else None
        return (input_cuda, weight_cuda, bias_cuda)
    
    def call_op(self, input_tuple):
        input_cuda, weight_cuda, bias_cuda = input_tuple
        return softplus_linear(input_cuda, weight_cuda, bias_cuda, beta=1, threshold=20)
    
    def get_gbps(self, input_tuple, runtime):
        input_tensor, weight, bias = input_tuple
        element_size = input_tensor.element_size()
        batch_size, in_features = input_tensor.shape
        out_features = weight.shape[0]
        
        # 计算总传输字节数
        input_bytes = input_tensor.numel() * element_size
        weight_bytes = weight.numel() * element_size
        bias_bytes = bias.numel() * element_size if bias is not None else 0
        linear_output_size = batch_size * out_features
        total_bytes = (input_bytes + weight_bytes + bias_bytes) + 3 * linear_output_size * element_size
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tuple, runtime):
        input_tensor, weight, bias = input_tuple
        batch_size, in_features = input_tensor.shape
        out_features = weight.shape[0]
        
        # 线性层FLOPS（矩阵乘法 + 偏置）
        matrix_mult_flops = 2 * in_features * out_features
        bias_flops = out_features if bias is not None else 0
        # Softplus FLOPS（假设每个元素6次操作）
        softplus_flops = 6 * out_features
        total_flops = matrix_mult_flops + bias_flops + softplus_flops
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
