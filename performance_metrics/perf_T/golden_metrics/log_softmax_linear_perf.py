import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.log_softmax_linear import log_softmax_linear
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('log_softmax_linear', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        batch_size = 32  # 固定batch_size
        
        # 测试不同规模的输入特征维度
        for i in range(4, 15):  # 调整i的范围控制测试规模
            in_features = 2 ** i
            out_features = 2 ** i
            input_tensor = torch.randn(batch_size, in_features, dtype=torch.float32)
            weight = torch.randn(out_features, in_features, dtype=torch.float32)
            bias = torch.randn(out_features, dtype=torch.float32)
            self.input_tensors.append((input_tensor, weight, bias))

    def to_cuda(self, input_tuple):
        input_tensor, weight, bias = input_tuple
        return (input_tensor.cuda(), weight.cuda(), bias.cuda())
    
    def call_op(self, input_tuple):
        input_tensor, weight, bias = input_tuple
        return log_softmax_linear(input_tensor, weight, bias, dim=-1, dtype=None)
    
    def get_gbps(self, input_tuple, runtime):
        input_tensor, weight, bias = input_tuple
        element_size = input_tensor.element_size()
        
        # 计算各张量的数据量
        input_bytes = input_tensor.numel() * element_size
        weight_bytes = weight.numel() * element_size
        bias_bytes = bias.numel() * element_size
        output_bytes = (input_tensor.shape[0] * weight.shape[0]) * element_size
        
        total_bytes = input_bytes + weight_bytes + bias_bytes + output_bytes * 5
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tuple, runtime):
        input_tensor, weight, bias = input_tuple
        batch = input_tensor.shape[0]
        in_fea = input_tensor.shape[1]
        out_fea = weight.shape[0]
        
        # 计算各阶段的浮点操作
        flops_matmul = 2 * batch * in_fea * out_fea      # 矩阵乘法
        flops_bias = batch * out_fea                     # 偏置加法
        flops_softmax = 3 * batch * out_fea              # log_softmax（exp、sum、log）
        
        total_flops = flops_matmul + flops_bias + flops_softmax
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
