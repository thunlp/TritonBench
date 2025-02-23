import sys
import os
import json

sys.path.append('/home/lishangzhan/triton/torch_performance/GPU_efficiency/output_DeepSeek-R1')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lu import lu
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('lu.py', dtype=dtype, is_backward=is_backward, **kwargs)
        if dtype is None:
            self.dtype = torch.float32  # 默认数据类型设为float32

    def get_input_tensors(self):
        self.input_tensors = []
        # 生成不同大小的方阵，例如从64x64到4096x4096
        for i in range(2, 13):  # 2^6=64 到 2^12=4096
            size = 128 * i
            input_tensor = torch.rand(size, size, dtype=self.dtype)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()

    def call_op(self, input_tensor):
        # 调用LU分解，启用部分选主元
        return lu(input_tensor, pivot=True)

    def get_gbps(self, input_tensor, runtime):
        # 总字节数 = 输入(A) + 输出(P, L, U)
        # 假设所有张量大小与输入相同（适用于方阵）
        total_bytes = 4 * input_tensor.numel() * input_tensor.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tensor, runtime):
        # LU分解的浮点运算次数约为(2/3) * n^3，其中n为矩阵行数
        n = input_tensor.size(0)
        flops = (2.0 / 3) * (n ** 3)
        TFLOPS = flops / (runtime / 1000) / 1e12
        return TFLOPS
    
    def run_benchmark(self):
        results = []
        for input_tensor_ in self.input_tensors:
            try:
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
            except Exception as e:
                print(f"Failed to run benchmark for input tensor. Error: {e}")
            input_tensor = None
        folder_path = "./results"
        file_name = self.op_name + ".json"
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'w', encoding='utf8') as f:
            json.dump(results, f, indent=4)


if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
