import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file
import quark
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase
from quark.torch.quantization import FP8E4M3PerChannelSpec
FP8_PER_CHANNEL_SPEC = FP8E4M3PerChannelSpec(is_dynamic=False, ch_axis=0).to_quantization_spec()
qspec=FP8_PER_CHANNEL_SPEC

from kernel import weight_dequant

def main(fp8_path, bf16_path):
    """
    Converts FP8 weights to BF16 and saves the converted weights.

    This function reads FP8 weights from the specified directory, converts them to BF16,
    and saves the converted weights to another specified directory. It also updates the
    model index file to reflect the changes.

    Args:
    fp8_path (str): The path to the directory containing the FP8 weights and model index file.
    bf16_path (str): The path to the directory where the converted BF16 weights will be saved.

    Raises:
    KeyError: If a required scale_inv tensor is missing for a weight.

    Notes:
    - The function assumes that the FP8 weights are stored in safetensor files.
    - The function caches loaded safetensor files to optimize memory usage.
    - The function updates the model index file to remove references to scale_inv tensors.
    """
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(bf16_path, exist_ok=True)
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"] # {'model.embed_tokens.weight': 'model-00001-of-000163.safetensors', 'model.layers.0.self_attn.q_a_proj.weight': 'model-00001-of-000163.safetensors'】
    
    # Cache for loaded safetensor files
    loaded_files = {}
    fp8_weight_names = []


    # Helper function to get tensor from the correct file
    def get_tensor(tensor_name):
        """
        Retrieves a tensor from the cached safetensor files or loads it from disk if not cached.

        Args:
            tensor_name (str): The name of the tensor to retrieve.

        Returns:
            torch.Tensor: The retrieved tensor.

        Raises:
            KeyError: If the tensor does not exist in the safetensor file.
        """
        file_name = weight_map[tensor_name]
        if file_name not in loaded_files:
            file_path = os.path.join(fp8_path, file_name)
            loaded_files[file_name] = load_file(file_path, device="cuda")
        return loaded_files[file_name][tensor_name]

    safetensor_files = list(glob(os.path.join(fp8_path, "*.safetensors")))
    safetensor_files.sort()
    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cuda")
        loaded_files[file_name] = current_state_dict
        
        new_state_dict = {}
        for weight_name, weight in current_state_dict.items():
            # we don't save weight_scale_inv here, just save weight.scale in following code
            if weight_name.endswith("_scale_inv"): 
                continue
            elif weight.element_size() == 1:  # FP8 weight with weight_scale_inv
                scale_inv_name = f"{weight_name}_scale_inv"
                quark_scale_name = f"{weight_name}_scale"
                try:
                    # Get scale_inv from the correct file
                    scale_inv = get_tensor(scale_inv_name)
                    fp8_weight_names.append(weight_name)
                    # DQ
                    bf_weight = weight_dequant(weight, scale_inv)
                    _weight_quantizer = FakeQuantizeBase.get_fake_quantize(qspec, device="cuda")
                    bf_weight = _weight_quantizer(bf_weight)
                    assert isinstance(bf_weight, torch.Tensor)
                    # Q
                    dtype = qspec.dtype.value
                    param = bf_weight
                    scale = _weight_quantizer.scale
                    zero_point = _weight_quantizer.zero_point
                    ch_axis = qspec.ch_axis
                    group_size = qspec.group_size
                    quant_min = -448
                    quant_max = 448
                    round_method = getattr(qspec.round_method, "value", None)
                    qscheme_str_name = getattr(qspec.qscheme, "value", None)
                    w_res = quark.torch.kernel.scaled_real_quantize(dtype, param, scale, zero_point, ch_axis, group_size, quant_min, quant_max, round_method, qscheme_str_name)
                    if scale_inv_name in weight_map: # clear weight_map[weight: *safetensors, scale: *safetensors]
                        # which_safe_file = weight_map[scale_inv_name]
                        # If the scale is obtained from another safe(s2), then delete the scale from s2 (achieved by scale_inv continue during traversal of s2);
                        # And in the weight_map, change the mapping from scale_inv and the safe file s2 to quark_scale and the current safe file.
                        weight_map.pop(scale_inv_name) 
                        weight_map[quark_scale_name] = file_name
                    new_state_dict[weight_name] = w_res
                    new_state_dict[quark_scale_name] = scale
                except KeyError:
                    print(f"Warning: Missing scale_inv tensor for {weight_name}, skipping conversion")
                    new_state_dict[weight_name] = weight
            else:
                new_state_dict[weight_name] = weight
                
        new_safetensor_file = os.path.join(bf16_path, file_name)
        save_file(new_state_dict, new_safetensor_file)
        
        # Memory management: keep only the 2 most recently used files
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
            torch.cuda.empty_cache()
    
    # After traversing all safe elements, proceed to create the weight_map.
    # Update model index
    new_model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    # Perform a secondary check to ensure scale_inv has been deleted.
    if 0: 
        for weight_name in fp8_weight_names:
            scale_inv_name = f"{weight_name}_scale_inv"
            if scale_inv_name in weight_map:
                print("warinng: there is a scale_inv")
                weight_map.pop(scale_inv_name)
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-fp8-hf-path", type=str, default="/mnt/raid0/pretrained_model/deepseek-ai/DeepSeek-V3.2-Exp")
    parser.add_argument("--output-fp8-hf-path-ptpc", type=str, default="/mnt/raid0/pretrained_model/deepseek-ai/DeepSeek-V3.2-Exp-ptpc")
    args = parser.parse_args()
    main(args.input_fp8_hf_path, args.output_fp8_hf_path_ptpc)