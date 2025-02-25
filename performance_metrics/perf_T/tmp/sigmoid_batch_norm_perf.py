import sys
import os
import json

sys.path.append('/home/lishangzhan/triton/torch_performance/GPU_efficiency/output_DeepSeek-R1_rag')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sigmoid_batch_norm import sigmoid_batch_norm
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('sigmoid_batch_norm', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):  # 测试不同规模的数据
            size = 2 ** i       # 总元素数为2^i
            batch_size = 32     # 固定batch_size
            channels = size // batch_size
            
            # 生成输入张量 (N, C)
            input_tensor = torch.randn(batch_size, channels, dtype=torch.float32)
            
            # 生成BatchNorm参数 (需要与输入通道数匹配)
            running_mean = torch.randn(channels)
            running_var = torch.abs(torch.randn(channels)) + 1e-5  # 保证方差为正
            weight = torch.randn(channels)
            bias = torch.randn(channels)
            
            self.input_tensors.append((
                input_tensor,
                running_mean,
                running_var,
                weight,
                bias
            ))

    def to_cuda(self, input_tuple):
        # 将所有参数转移到CUDA
        return tuple(t.cuda() if isinstance(t, torch.Tensor) else t for t in input_tuple)
    
    def call_op(self, input_tuple):
        # 解包参数并调用算子
        input_tensor, running_mean, running_var, weight, bias = input_tuple
        return sigmoid_batch_norm(
            input=input_tensor,
            running_mean=running_mean,
            running_var=running_var,
            weight=weight,
            bias=bias,
            training=False  # 测试模式不更新running stats
        )
    
    def get_gbps(self, input_tuple, runtime):
        # 计算总数据吞吐量（GB/s）
        input_tensor, running_mean, running_var, weight, bias = input_tuple
        element_size = input_tensor.element_size()
        
        # 输入数据量
        input_bytes = input_tensor.numel() * element_size
        running_mean_bytes = running_mean.numel() * element_size
        running_var_bytes = running_var.numel() * element_size
        weight_bytes = weight.numel() * element_size
        bias_bytes = bias.numel() * element_size
        
        # 输出数据量（与输入同shape）
        output_bytes = input_tensor.numel() * element_size
        
        total_bytes = input_bytes + running_mean_bytes + running_var_bytes + weight_bytes + bias_bytes + output_bytes * 3
        return total_bytes / (runtime / 1000) / 1e9  # GB/s

    def get_tflops(self, input_tuple, runtime):
        # 计算理论计算量（TFLOPs）
        input_tensor, _, _, _, _ = input_tuple
        # 每个元素的运算次数分解：
        # BatchNorm: (x-mean)/sqrt(var+eps)*gamma + beta → 6次运算
        # Sigmoid: 1/(1+exp(-x)) → 4次运算
        flops_per_element = 10
        total_flops = input_tensor.numel() * flops_per_element
        return total_flops / (runtime / 1000) / 1e12  # TFLOP/s

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
