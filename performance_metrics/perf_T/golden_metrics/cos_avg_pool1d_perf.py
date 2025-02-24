import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.cos_avg_pool1d import cos_avg_pool1d
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('cos_avg_pool1d', dtype=dtype, is_backward=is_backward, **kwargs)
        self.kernel_size = 3  # 固定kernel_size为3

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 24):  # 生成不同规模的输入
            W = 128 * i
            input_shape = (32, 32, W)  # (minibatch, in_channels, iW)
            input_tensor = torch.rand(*input_shape, dtype=self.dtype or torch.float32)
            
            # 预计算输出形状
            with torch.no_grad():
                output = cos_avg_pool1d(input_tensor, kernel_size=self.kernel_size)
            self.input_tensors.append((input_tensor, output.shape))

    def to_cuda(self, input_tensor):
        input_tensor_, _ = input_tensor
        return (input_tensor_.cuda(), _)
    
    def call_op(self, input_tensor):
        input_tensor_, _ = input_tensor
        return cos_avg_pool1d(input_tensor_, kernel_size=self.kernel_size)
    
    def get_gbps(self, input_tensor, runtime):
        # 根据预存的输出形状计算带宽
        # idx = self.input_tensors.index(input_tensor)
        # output_shape = self.output_shapes[idx]
        input_tensor_, output_shape = input_tensor
        output_numel = torch.Size(output_shape).numel()
        
        element_size = input_tensor_.element_size()
        total_bytes = (input_tensor_.numel() + output_numel) * element_size + 2 * input_tensor_.numel() * element_size
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        # 计算总浮点运算量（cos + avg_pool）
        # idx = self.input_tensors.index(input_tensor)
        # output_shape = self.output_shapes[idx]
        input_tensor_, output_shape = input_tensor
        output_numel = torch.Size(output_shape).numel()
        
        flops_cos = input_tensor_.numel()  # 每个元素一个cos运算
        flops_pool = output_numel * self.kernel_size  # 每个输出元素对应kernel_size次运算
        total_flops = flops_cos + flops_pool
        
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
