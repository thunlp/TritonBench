# TritonBench

TritonBench features two distinct channels: **TritonBench-G** and **TritonBench-T**, each with its own evaluation framework. For detailed information, refer to the paper [TRITONBENCH: Benchmarking Large Language Model Capabilities for Generating Triton Operators](https://arxiv.org/pdf/2502.14752).

## Data
- **TritonBench-G** offers two versions of Alpaca-format instructions: 
  - Simple instruction: `TritonBench_G_simp_alpac_v1.json`
  - Complex instruction: `TritonBench_G_comp_alpac_v1.json`
- It also includes executable folders (`TritonBench_G_v1`) and associated statistics (`TritonBench_G_v1.json`).
- **TritonBench-T** offers two versions of Alpaca-format instructions: 
  - Simple instruction: `TritonBench_T_simp_alpac_v1.json`
  - Complex instruction: `TritonBench_T_comp_alpac_v1.json`
- It also includes executable folders (`TritonBench_T_v1`) and associated statistics (`TritonBench_T_v1.json`).
- Additionally, there are two sets of filtered GitHub data:
  - `train_crawl.json` (4024 entries) – de-duplicated using BERT score similarity.
  - `train_synth.json` (4133 entries) – data synthesized using Jiuci.
- The combined 8k dataset can be used for **RAG** (Retrieval-Augmented Generation).

## LLM Generated
We also provide the output results from all major models used in the paper.

## Python Environment
- `triton = 3.1.0`
- `torch >= 2.5.1`
- After installation, update the `py_interpreter` paths in `eval_G` and `eval_T`.

## Evaluation Process
### TritonBench-G
1. **Code Similarity Evaluation**: First, use **CodeBLEU** to evaluate code similarity. For detailed instructions, refer to `../readme_4similarity.md`.
2. **Execution Accuracy**: 
    - Run `0_call_acc.py` with the following command:
    ```bash
    0_call_acc.py --source source/path/or/folder --target target/path/or/folder --GPUs [0,1,2,3]
    ```
    - Multiple GPUs can accelerate the execution.
3. **Execution Performance**: 
    - Run `1_exe_acc.py` with:
    ```bash
    1_exe_acc.py --folder root/of/multiple/folders/or/folder --GPUs [0,1,2,3]
    ```
4. **Efficiency**: 
    - Finally, run `2_efficiency.py`. 李尚展补全（注意argparse传参替换掉lijianling路径）

### TritonBench-T
For **TritonBench-T**, there is no code similarity evaluation. Only call accuracy, execution accuracy, and speedup are assessed. The process is similar:
1. Run `0_call_acc.py` as above:
    ```bash
    0_call_acc.py --source source/path/or/folder --target target/path/or/folder --GPUs [0,1,2,3]
    ```
2. Run `1_exe_acc.py` with the appropriate folders and GPUs:
    ```bash
    1_exe_acc.py --folder root/of/multiple/folders/or/folder --GPUs [0,1,2,3]
    ```
3. Run `2_efficiency.py` 李尚展补全（注意argparse传参替换掉lijianling路径）

**Note**: Ensure that accuracy and efficiency evaluations are performed sequentially.