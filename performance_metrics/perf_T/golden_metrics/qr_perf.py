import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TorchBench_v1.qr import qr
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('qr', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        # 生成从2^8到2^12的方阵作为测试输入
        for i in range(4, 13):
            size = 2 ** i
            input_tensor = torch.rand((size, size), dtype=self.dtype or torch.float32)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        # 将输入张量转移到CUDA设备
        return input_tensor.cuda()
    
    def call_op(self, input_tensor):
        # 调用QR分解算子，使用默认的reduced模式
        return qr(input_tensor, mode='reduced')
    
    def get_gbps(self, input_tensor, runtime):
        # 计算总数据量：输入矩阵 + Q矩阵 + R矩阵
        total_bytes = 3 * input_tensor.numel() * input_tensor.element_size()
        # 转换为GB/s（千兆字节每秒）
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        # 获取矩阵维度（假设为方阵）
        n = input_tensor.size(0)
        # QR分解的浮点运算次数（4/3*n^3）
        flops = (4.0 / 3.0) * (n ** 3)
        # 转换为TFLOPS（万亿次浮点运算每秒）
        TFLOPS = flops / (runtime / 1000) / 1e12
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
