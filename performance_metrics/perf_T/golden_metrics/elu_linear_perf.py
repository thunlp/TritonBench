import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.elu_linear import elu_linear
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('elu_linear', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        # 测试不同输入尺寸，从2^8到2^16以控制内存使用
        for i in range(8, 17):
            batch_size = 1024
            in_features = 2 ** i
            out_features = 256  # 固定输出特征数
            # 生成输入张量、权重和偏置
            input_tensor = torch.randn(batch_size, in_features, dtype=self.dtype)
            weight = torch.randn(out_features, in_features, dtype=self.dtype)
            bias = torch.randn(out_features, dtype=self.dtype)
            self.input_tensors.append((input_tensor, weight, bias))

    def to_cuda(self, input_tuple):
        # 将每个张量转移到CUDA
        input_tensor, weight, bias = input_tuple
        return (input_tensor.cuda(), weight.cuda(), bias.cuda())
    
    def call_op(self, input_tuple):
        # 调用算子并返回结果
        input_tensor, weight, bias = input_tuple
        return elu_linear(input_tensor, weight, bias)

    def get_gbps(self, input_tuple, runtime):
        # 计算总数据传输量（GB/s）
        input_tensor, weight, bias = input_tuple
        batch_size, in_features = input_tensor.shape
        out_features = weight.shape[0]
        element_size = input_tensor.element_size()
        
        # 输入、权重、偏置的读取 + 输出的写入
        total_bytes = (input_tensor.numel() + weight.numel() + bias.numel() + 
                       batch_size * out_features) * element_size * 2
        return total_bytes / (runtime / 1000) / 1e9  # 转换为GB/s

    def get_tflops(self, input_tuple, runtime):
        # 计算浮点运算量（TFLOPs）
        input_tensor, weight, bias = input_tuple
        batch_size, in_features = input_tensor.shape
        out_features = weight.shape[0]
        
        # 线性层矩阵乘法: 2 * B * I * O
        flops_linear = 2 * batch_size * in_features * out_features
        # 线性层偏置加法: B * O 
        flops_bias = batch_size * out_features
        # ELU激活: 3 * B * O (假设每个元素3次操作)
        flops_elu = 3 * batch_size * out_features
        
        total_flops = flops_linear + flops_bias + flops_elu
        return total_flops / (runtime / 1000) / 1e12  # 转换为TFLOP/s


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
