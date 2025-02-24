import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.fused_hstack_div import fused_hstack_div
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_hstack_div', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 28):
            num_elements = 2 ** i
            # 生成两个一维张量，每个包含num_elements元素
            tensor1 = torch.rand(num_elements, dtype=self.dtype)
            tensor2 = torch.rand(num_elements, dtype=self.dtype)
            tensors = [tensor1, tensor2]
            divisor = 2.0  # 使用标量作为除数
            self.input_tensors.append((tensors, divisor))
    
    def to_cuda(self, input_case):
        tensors, divisor = input_case
        # 转移张量列表中的每个张量到CUDA
        cuda_tensors = [t.cuda() for t in tensors]
        # 如果divisor是张量，也需转移到CUDA
        if isinstance(divisor, torch.Tensor):
            cuda_divisor = divisor.cuda()
        else:
            cuda_divisor = divisor
        return (cuda_tensors, cuda_divisor)
        
    def call_op(self, input_case):
        tensors, divisor = input_case
        return fused_hstack_div(tensors, divisor)

    def get_gbps(self, input_case, runtime):
        tensors, divisor = input_case
        # 计算输入总元素数和输出元素数
        input_elements = sum(t.numel() for t in tensors)
        output_elements = torch.hstack(tensors).numel()
        element_size = tensors[0].element_size()
        # 总字节数 = (输入 + 输出) * 元素大小
        total_bytes = (input_elements + output_elements * 3) * element_size
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_case, runtime):
        tensors, divisor = input_case
        # 每个输出元素对应一次除法操作
        output_elements = torch.hstack(tensors).numel()
        TFLOPS = output_elements / (runtime / 1000) / 1e12
        return TFLOPS

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
                "input_size": [input_tensor[0][0].shape],
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
