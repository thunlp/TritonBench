import sys
import os
import json

sys.path.append('/home/lishangzhan/triton/torch_performance/GPU_efficiency/output_DeepSeek-R1_rag')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dropout_sigmoid_linear import dropout_sigmoid_linear
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('dropout_sigmoid_linear', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        batch_size = 1024
        out_features = 256  # 固定输出维度
        for exp in range(8, 16):  # in_features从2^8到2^14
            in_features = 2 ** exp
            input_tensor = torch.randn(batch_size, in_features, dtype=self.dtype)
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
        input_tensor, weight, bias = input_tuple
        return dropout_sigmoid_linear(input_tensor, weight, bias, p=0.5, training=True, inplace=False)
    
    def get_gbps(self, input_tuple, runtime):
        input_tensor, weight, bias = input_tuple
        M, K = input_tensor.shape
        N, _ = weight.shape
        
        element_size = input_tensor.element_size()
        input_size = input_tensor.numel() * element_size
        weight_size = weight.numel() * element_size
        bias_size = bias.numel() * element_size if bias is not None else 0
        output_size = M * N * element_size
        
        total_bytes = (input_size + weight_size + bias_size + output_size) * 3
        GBPS = total_bytes / (runtime / 1000) / 1e9  # 转换为GB/s
        return GBPS
    
    def get_tflops(self, input_tuple, runtime):
        input_tensor, weight, bias = input_tuple
        M, K = input_tensor.shape
        N, _ = weight.shape
        
        # 计算各部分的FLOPS
        linear_flops = 2 * M * N * K + M * N  # 矩阵乘法 + bias加法
        sigmoid_flops = 4 * M * N              # 每个元素4次运算
        dropout_flops = 1 * M * N              # 每个元素1次运算
        
        total_flops = linear_flops + sigmoid_flops + dropout_flops
        TFLOPS = total_flops / (runtime / 1000) / 1e12  # 转换为TFLOPS
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
