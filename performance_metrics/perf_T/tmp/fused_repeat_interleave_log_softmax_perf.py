import sys
import os
import json

sys.path.append('/home/lishangzhan/triton/torch_performance/GPU_efficiency/output_DeepSeek-R1_rag')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fused_repeat_interleave_log_softmax import fused_repeat_interleave_log_softmax
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('fused_repeat_interleave_log_softmax', dtype=dtype, is_backward=is_backward, **kwargs)
        self.repeats = 2  # 固定重复次数
        self.dim = 0      # 固定操作维度

    def get_input_tensors(self):
        self.input_tensors = []
        # 生成不同大小的1D输入张量（防止显存溢出）
        for i in range(12, 24):  # 调整范围到2^12 ~ 2^23
            size = 2 ** i
            input_tensor = torch.rand(size, dtype=torch.float32)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()
    
    def call_op(self, input_tensor):
        # 调用算子并传入固定参数
        return fused_repeat_interleave_log_softmax(input_tensor, 
                                                 repeats=self.repeats, 
                                                 dim=self.dim)
    
    def get_gbps(self, input_tensor, runtime):
        # 计算总数据量（包含中间张量）
        element_size = input_tensor.element_size()
        input_numel = input_tensor.numel()
        total_bytes = input_numel * element_size * (1 + 3 * self.repeats)
        GBPS = total_bytes / (runtime / 1000) / 1e9  # 转换毫秒到秒
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        # 仅计算log_softmax的浮点运算量
        flops = 3 * input_tensor.numel() * self.repeats  # 3 FLOPs per element
        TFLOPS = flops / (runtime / 1000) / 1e12  # 转换毫秒到秒
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
