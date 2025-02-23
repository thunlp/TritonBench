import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.fused_hardsigmoid_batch_norm import fused_hardsigmoid_batch_norm
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_hardsigmoid_batch_norm', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):
            num_elements = 2 ** i
            # 构造四维输入张量 (N, C, H, W)
            c = 64  # 通道数
            n = 1
            hw = num_elements // (n * c)
            h = int(hw ** 0.5)
            w = h if h * h >= hw else h + 1
            # 确保张量元素总数足够大
            while n * c * h * w < num_elements:
                w += 1
            x = torch.randn(n, c, h, w, dtype=torch.float32)
            running_mean = torch.randn(c)
            running_var = torch.randn(c).abs() + 1e-5  # 方差需为正
            weight = torch.randn(c)
            bias = torch.randn(c)
            # 固定非训练模式以保证结果稳定性
            input_tuple = (x, running_mean, running_var, weight, bias, False, 0.1, 1e-5, False)
            self.input_tensors.append(input_tuple)
    
    def to_cuda(self, input_tuple):
        # 转移所有张量至CUDA
        x, rm, rv, w, b, train, mom, eps, inp = input_tuple
        return (
            x.cuda(), rm.cuda(), rv.cuda(), 
            w.cuda() if w is not None else None, 
            b.cuda() if b is not None else None, 
            train, mom, eps, inp
        )
    
    def call_op(self, input_tuple):
        # 解包参数并调用融合算子
        return fused_hardsigmoid_batch_norm(*input_tuple)
    
    def get_gbps(self, input_tensor, runtime):
        # 计算数据吞吐量 (GB/s)
        x = input_tensor[0]
        total_bytes = 4 * x.numel() * x.element_size()
        return total_bytes / (runtime / 1000) / 1e9
    
    def get_tflops(self, input_tensor, runtime):
        # 计算计算吞吐量 (TFLOP/s)
        x = input_tensor[0]
        flops_per_element = 7  # BN(4) + Hardsigmoid(3)
        total_flops = x.numel() * flops_per_element
        return total_flops / (runtime / 1000) / 1e12


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
