import sys
import os
import json

sys.path.append('/home/lishangzhan/triton/torch_performance/GPU_efficiency/output_DeepSeek-R1')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from broadcast_tensors import broadcast_tensors
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('broadcast_tensors', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for exp in range(10, 24):  # 控制规模避免内存溢出
            size = 2 ** exp
            x = torch.rand((1, size), dtype=torch.float32)
            y = torch.rand((size, 1), dtype=torch.float32)
            self.input_tensors.append((x, y))
    
    def to_cuda(self, input_tensor):
        x, y = input_tensor
        return (x.cuda(), y.cuda())
        
    def call_op(self, input_tensor):
        x, y = input_tensor
        return broadcast_tensors(x, y)
    
    def get_gbps(self, input_tensor, runtime):
        return 0
    
    def get_tflops(self, input_tensor, runtime):
        # 广播操作无浮点运算，TFLOPS为0
        return 0.0


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
