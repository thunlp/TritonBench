import sys
import os
import json

sys.path.append('/home/lishangzhan/triton/torch_performance/GPU_efficiency/output_DeepSeek-R1_rag')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sigmoid_argmax import sigmoid_argmax
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('sigmoid_argmax', dtype=dtype, is_backward=is_backward, **kwargs)

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 29):  # 测试从2^12到2^28的不同规模
            size = 2 ** i
            # 生成二维张量模拟典型使用场景
            rows = 2 ** (i // 2)
            cols = size // rows
            assert rows * cols == size, f"Dimension error: {rows}*{cols} != {size}"
            input_tensor = torch.rand((rows, cols), dtype=self.dtype or torch.float32)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        return input_tensor.cuda()
    
    def call_op(self, input_tensor):
        # 典型使用场景：沿着最后一个维度做argmax
        return sigmoid_argmax(input_tensor, dim=-1, keepdim=False)
    
    def get_gbps(self, input_tensor, runtime):
        # 计算总数据吞吐量：输入+中间结果（sigmoid输出）+中间结果读取
        N = input_tensor.numel()
        element_size = input_tensor.element_size()
        total_bytes = 3 * N * element_size  # 输入读 + 中间写 + 中间读
        return total_bytes / (runtime / 1000) / 1e9
    
    def get_tflops(self, input_tensor, runtime):
        # 每个元素的sigmoid操作需要3次浮点运算（exp/加法/倒数）
        N = input_tensor.numel()
        return (3 * N) / (runtime / 1000) / 1e12
    
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
