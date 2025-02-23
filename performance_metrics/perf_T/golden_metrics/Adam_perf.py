import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.Adam import Adam
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('Adam', dtype=dtype, is_backward=is_backward, **kwargs)
        self.lr = 0.001
        self.betas = (0.9, 0.999)
        self.eps = 1e-8
        self.weight_decay = 0

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):
            size = 2 ** i
            param = torch.rand(size, dtype=self.dtype, requires_grad=True)
            param.grad = torch.rand_like(param)  # 预先生成梯度
            self.input_tensors.append([param])    # 参数列表作为输入

    def to_cuda(self, input_params):
        for param in input_params:
            param.data = param.cuda()
            if param.grad is not None:
                param.grad.data = param.grad.cuda()
        return input_params

    def call_op(self, input_params):
        optimizer = Adam(input_params, 
                        lr=self.lr,
                        betas=self.betas,
                        eps=self.eps,
                        weight_decay=self.weight_decay)
        optimizer.step()

    def get_gbps(self, input_params, runtime):
        total_bytes = 0
        for param in input_params:
            numel = param.numel()
            # 内存访问: 读(param+grad+m+v) + 写(param+m+v)
            total_bytes += numel * (4*4 + 3*4)  # 28 bytes/element
        return total_bytes / (runtime / 1000) / 1e9

    def get_tflops(self, input_params, runtime):
        total_flops = 0
        for param in input_params:
            numel = param.numel()
            # 估算每个参数元素的浮点操作次数（约10次）
            total_flops += numel * 10
        return total_flops / (runtime / 1000) / 1e12


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
