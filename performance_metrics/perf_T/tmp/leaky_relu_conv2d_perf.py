import sys
import os
import json

sys.path.append('/home/lishangzhan/triton/torch_performance/GPU_efficiency/output_DeepSeek-R1')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 引入 leaky_relu_conv2d 算子
from leaky_relu_conv2d import leaky_relu_conv2d
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('leaky_relu_conv2d', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        # 定义不同size的输入张量
        self.input_tensors = []
        for i in range(2, 12):  # 从2^12到2^27大小的张量
            size = 128 * i
            input_tensor = torch.rand(1, 3, size, size, dtype=torch.float32)  # 输入tensor大小，假设是(64, 64)的图片
            weight_tensor = torch.rand(6, 3, 3, 3, dtype=torch.float32)  # 卷积核大小为(16, 3, 3, 3)
            bias_tensor = torch.rand(6, dtype=torch.float32)  # 偏置
            self.input_tensors.append((input_tensor, weight_tensor, bias_tensor))

    def to_cuda(self, input_tensor):
        # 将输入张量和权重、偏置移到CUDA
        input_tensor, weight_tensor, bias_tensor = input_tensor
        return input_tensor.cuda(), weight_tensor.cuda(), bias_tensor.cuda()

    def call_op(self, input_tensor):
        # 调用leaky_relu_conv2d算子
        input_tensor, weight_tensor, bias_tensor = input_tensor
        return leaky_relu_conv2d(input_tensor, weight_tensor, bias_tensor)

    def get_gbps(self, input_tensor, runtime):
        # 计算GBPS
        input_tensor, weight_tensor, bias_tensor = input_tensor
        # 输入和输出的字节数
        total_bytes = (input_tensor.numel() + weight_tensor.numel() + bias_tensor.numel()) * input_tensor.element_size() * 4
        GBPS = total_bytes / (runtime / 1000) / 1e9  # runtime是毫秒，转换为秒
        return GBPS

    def get_tflops(self, input_tensor, runtime):
        # 计算TFLOPS
        input_tensor, weight_tensor, bias_tensor = input_tensor
        FLOPS = input_tensor.numel() * weight_tensor.numel()  # 计算FLOPS
        TFLOPS = FLOPS / (runtime / 1000) / 1e12  # runtime是毫秒，转换为秒
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
