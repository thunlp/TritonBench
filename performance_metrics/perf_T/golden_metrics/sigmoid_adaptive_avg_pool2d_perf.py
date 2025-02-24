import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.sigmoid_adaptive_avg_pool2d import sigmoid_adaptive_avg_pool2d
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, output_size=(1,1), **kwargs):
        super().__init__('sigmoid_adaptive_avg_pool2d', dtype=dtype, is_backward=is_backward, **kwargs)
        self.output_size = output_size  # 固定output_size参数

    def get_input_tensors(self):
        self.input_tensors = []
        # 生成不同尺寸的四维张量 (batch, channels, height, width)
        for i in range(8, 18):  # 调整范围控制输入尺寸
            batch = 2 ** (i % 4)
            channels = 2 ** (i % 4)
            h = 2 ** (i // 2)
            w = 2 ** (i // 2)
            input_tensor = torch.rand(batch, channels, h, w, dtype=torch.float32)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()  # 张量迁移到CUDA
    
    def call_op(self, input_tensor):
        return sigmoid_adaptive_avg_pool2d(input_tensor, self.output_size)  # 调用目标算子

    def get_gbps(self, input_tensor, runtime):
        # 计算内存带宽吞吐量
        input_bytes = input_tensor.numel() * input_tensor.element_size()
        output_numel = (input_tensor.size(0) * input_tensor.size(1) 
                        * self.output_size[0] * self.output_size[1])
        output_bytes = output_numel * input_tensor.element_size()
        total_bytes = input_bytes + output_bytes
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tensor, runtime):
        # 计算理论计算吞吐量
        output_numel = (input_tensor.size(0) * input_tensor.size(1)
                       * self.output_size[0] * self.output_size[1])
        H, W = input_tensor.shape[2], input_tensor.shape[3]
        kh = H // self.output_size[0]  # 池化窗口高度
        kw = W // self.output_size[1]  # 池化窗口宽度
        
        # 每个输出元素的FLOPs = 池化窗口元素数(加法) + 1(除法) + 3(sigmoid)
        flops_per_element = kh * kw + 1 + 3
        total_flops = output_numel * flops_per_element
        
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
