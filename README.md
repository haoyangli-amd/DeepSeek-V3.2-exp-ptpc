# quantize deepseekv3.2-exp by QDQ safetensors with quark

## Installation

To get started, install:

```bash
  pip install amd-quark
```

## Quickstart

### 1) Execute the following script

```bash
  python3 DeepSeek-V3.2-exp-ptpc/inference/dpsk3.2exp_2_quark_ptpc.py --input_fp8_hf_path path_a  --output_fp8_hf_path_ptpc path_b
```

### 2) Modify and move config.json to the folder where the exported model is located.

```bash
  cp DeepSeek-V3.2-exp-ptpc/config.json path_b
```