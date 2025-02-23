import sys
import os
import json

sys.path.append('/home/lishangzhan/triton/torch_performance/GPU_efficiency/output_DeepSeek-R1')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fused_pairwise_distance_adaptive_avg_pool2d import fused_pairwise_distance_adaptive_avg_pool2d
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_pairwise_distance_adaptive_avg_pool2d', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 14):
            hw = 2 ** i
            x1 = torch.randn(16, 3, hw, hw, dtype=torch.float32)
            x2 = torch.randn(16, 3, hw, hw, dtype=torch.float32)
            output_size = (hw // 2, hw // 2)  # 池化到原尺寸的一半
            self.input_tensors.append((x1, x2, output_size))
    
    def to_cuda(self, input_tuple):
        x1, x2, output_size = input_tuple
        return (x1.cuda(), x2.cuda(), output_size)
        
    def call_op(self, input_tuple):
        x1, x2, output_size = input_tuple
        return fused_pairwise_distance_adaptive_avg_pool2d(x1, x2, output_size)
    
    def get_gbps(self, input_tuple, runtime):
        x1, x2, output_size = input_tuple
        # 计算输入和输出的总字节数
        x1_bytes = x1.numel() * x1.element_size()
        x2_bytes = x2.numel() * x2.element_size()
        B = x1.size(0)
        dist_bytes = B * x1.element_size()  # 输出尺寸为(B,)
        total_bytes = x1_bytes + x2_bytes + dist_bytes + x1_bytes / 4 * 6
        # 转换为GB/s
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tuple, runtime):
        x1, x2, output_size = input_tuple
        B, C, H, W = x1.shape
        # 解析池化后的尺寸
        if isinstance(output_size, int):
            H_prime, W_prime = output_size, output_size
        else:
            H_prime, W_prime = output_size
        
        # 计算池化操作的FLOPs（两次池化）
        pool_flops = 2 * B * C * (H * W + H_prime * W_prime)
        # 计算差值的FLOPs
        diff_flops = B * C * H_prime * W_prime
        # 计算范数的FLOPs（平方、求和、开根）
        norm_flops = B * 2 * C * H_prime * W_prime
        total_flops = pool_flops + diff_flops + norm_flops
        # 转换为TFLOPS
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
