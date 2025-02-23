import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.grid_sample import grid_sample
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('grid_sample', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        # 生成4D输入和对应grid
        for i in range(8, 12):  # 调整范围控制测试规模
            H = W = 2 ** i      # 输入分辨率
            C = 3               # 通道数
            N = 16               # batch size
            
            # 生成输入张量
            input_tensor = torch.randn(N, C, H, W, dtype=torch.float32)
            
            # 生成对应grid（输出分辨率设为输入的1/2）
            H_out = H // 2
            W_out = W // 2
            grid_tensor = torch.rand(N, H_out, W_out, 2, dtype=torch.float32)
            grid_tensor = grid_tensor * 2 - 1  # 转换到[-1, 1]范围
            
            self.input_tensors.append((input_tensor, grid_tensor))

    def to_cuda(self, input_tensor_tuple):
        input_tensor, grid_tensor = input_tensor_tuple
        return (input_tensor.cuda(), grid_tensor.cuda())
    
    def call_op(self, input_tensor_tuple):
        input_tensor, grid_tensor = input_tensor_tuple
        return grid_sample(input_tensor, grid_tensor)

    def get_gbps(self, input_tensor_tuple, runtime):
        input_tensor, grid_tensor = input_tensor_tuple
        element_size = input_tensor.element_size()  # 假设input和grid类型一致
        
        # 输入数据量
        input_bytes = input_tensor.numel() * element_size
        grid_bytes = grid_tensor.numel() * element_size
        
        # 输出数据量（根据grid形状计算）
        N, C = input_tensor.shape[0], input_tensor.shape[1]
        H_out, W_out = grid_tensor.shape[1], grid_tensor.shape[2]
        output_bytes = N * C * H_out * W_out * element_size
        
        total_bytes = input_bytes + grid_bytes + output_bytes
        return total_bytes / (runtime / 1000) / 1e9  # 转换为GB/s

    def get_tflops(self, input_tensor_tuple, runtime):
        input_tensor, grid_tensor = input_tensor_tuple
        N, C = input_tensor.shape[0], input_tensor.shape[1]
        H_out, W_out = grid_tensor.shape[1], grid_tensor.shape[2]
        
        # 假设每个输出元素需要7次浮点运算（双线性插值）
        flops_per_sample = 7
        total_flops = N * C * H_out * W_out * flops_per_sample
        
        return total_flops / (runtime / 1000) / 1e12  # 转换为TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
