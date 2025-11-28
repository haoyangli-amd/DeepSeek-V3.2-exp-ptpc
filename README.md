# quantize deepseekv3.2-exp by QDQ safetensors with quark

## Installation

To get started, install:

```bash
  pip install amd-quark
```

## Quickstart

### 1) Execute the following script

```bash
  python3 DeepSeek-V3.2-exp-ptpc/inference/dpsk3.2exp_2_quark_ptpc.py --input-fp8-hf-path path_a  --output-fp8-hf-path-ptpc path_b
```

### 2) copy *.json(model.safetensors.index.json will be regenerated. Please be careful not to overwrite it.) and *py from the original model folder to the folder where the exported model is located.

### 3) Modify and copy config.json to the folder where the exported model is located.

```bash
  cp DeepSeek-V3.2-exp-ptpc/config.json path_b
```

