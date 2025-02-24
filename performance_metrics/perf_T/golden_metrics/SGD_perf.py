import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.SGD import SGD
from performance_utils import Performance_Metrics, do_bench_config

import torch
import torch.nn as nn
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('SGD.py', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 24):  # Reduced range to avoid memory issues
            input_size = 2 ** i
            batch_size = 1
            model = nn.Linear(input_size, 10)  # Model with input_size features and 10 outputs
            input_tensor = torch.randn(batch_size, input_size)
            target = torch.randint(0, 10, (batch_size,))  # Random target classes
            loss_fn = nn.CrossEntropyLoss()
            self.input_tensors.append((model, input_tensor, target, loss_fn))
    
    def to_cuda(self, input_tuple):
        model, input_tensor, target, loss_fn = input_tuple
        return (model.cuda(), input_tensor.cuda(), target.cuda(), loss_fn)
        
    def call_op(self, input_tuple):
        model, input_tensor, target, loss_fn = input_tuple
        return SGD(model, input_tensor, target, loss_fn)

    def get_gbps(self, input_tuple, runtime):
        model, input_tensor, target, loss_fn = input_tuple
        # Calculate data transfer for input and output
        input_bytes = input_tensor.numel() * input_tensor.element_size()
        output = model(input_tensor)
        output_bytes = output.numel() * output.element_size()
        
        # Calculate parameter-related memory access (4 accesses per parameter)
        params = sum(p.numel() for p in model.parameters())
        param_bytes = 4 * params * input_tensor.element_size()
        
        total_bytes = input_bytes + output_bytes + param_bytes
        return total_bytes / (runtime / 1000) / 1e9
    
    def get_tflops(self, input_tuple, runtime):
        model, input_tensor, target, loss_fn = input_tuple
        B, N = input_tensor.shape
        output = model(input_tensor)
        C = output.shape[-1]
        params = sum(p.numel() for p in model.parameters())
        
        # Forward FLOPs: linear layer + bias
        flops_forward = B * N * C * 2 + B * C
        
        # Backward FLOPs (approx 2x forward)
        flops_backward = 2 * flops_forward
        
        # Update FLOPs (5 per parameter)
        flops_update = 5 * params
        
        total_flops = flops_forward + flops_backward + flops_update
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
                "input_size": [input_tensor[1].shape],
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
