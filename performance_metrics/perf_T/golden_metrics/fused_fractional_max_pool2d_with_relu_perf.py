import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.fused_fractional_max_pool2d_with_relu import fused_fractional_max_pool2d_with_relu
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_fractional_max_pool2d_with_relu', dtype=dtype, is_backward=is_backward, **kwargs)
        self.kernel_size = (3, 3)
        self.output_ratio = (0.5, 0.5)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 13):
            hw = 2 ** i
            dtype = self.dtype if self.dtype is not None else torch.float32
            input_tensor = torch.rand(16, 3, hw, hw, dtype=dtype)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()
    
    def call_op(self, input_tensor):
        return fused_fractional_max_pool2d_with_relu(
            input_tensor,
            kernel_size=self.kernel_size,
            output_ratio=self.output_ratio,
            return_indices=False
        )
    
    def get_gbps(self, input_tensor, runtime):
        element_size = input_tensor.element_size()
        H, W = input_tensor.shape[2], input_tensor.shape[3]
        H_out = int(H * self.output_ratio[0])
        W_out = int(W * self.output_ratio[1])
        
        input_bytes = input_tensor.numel() * element_size
        output_bytes = (input_tensor.shape[0] * input_tensor.shape[1] 
                        * H_out * W_out) * element_size
        
        total_bytes = input_bytes * 3 + output_bytes
        return total_bytes / (runtime / 1000) / 1e9

    def get_tflops(self, input_tensor, runtime):
        H, W = input_tensor.shape[2], input_tensor.shape[3]
        H_out = int(H * self.output_ratio[0])
        W_out = int(W * self.output_ratio[1])
        
        flops_relu = input_tensor.numel()
        
        output_elements = input_tensor.shape[0] * input_tensor.shape[1] * H_out * W_out
        flops_pool = output_elements * self.kernel_size[0] * self.kernel_size[1]
        
        total_flops = flops_relu + flops_pool
        return total_flops / (runtime / 1000) / 1e12
    
    def run_benchmark(self):
        results = []
        for input_tensor_ in self.input_tensors:
            input_tensor = self.to_cuda(input_tensor_)
            # print(input_tensor)
            op = lambda : self.call_op(input_tensor)
            ms = self.get_runtime(op)
            gbps = self.get_gbps(input_tensor, ms)
            tflops = self.get_tflops(input_tensor, ms)
            result = {
                "input_size": [input_tensor.shape],
                "ms": ms,
                "GB/s": gbps,
                "TFLOPS": tflops
            }
            print(result)
            results.append(result)
            input_tensor = None
        folder_path = "/home/lishangzhan/triton/torch_performance/results"
        file_name = self.op_name + ".json"
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'w', encoding='utf8') as f:
            json.dump(results, f, indent=4)

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
