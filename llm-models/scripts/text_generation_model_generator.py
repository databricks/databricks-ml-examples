import argparse
from dataclasses import dataclass
from huggingface_hub import model_info
import os
from typing import Any, Dict
import yaml
import logging
import sys

from template import TemplateManager

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
_logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
_logger.addHandler(handler)

_SECTIONS_PACKAGE = "./scripts/templates/"

FINE_TUNE_EXAMPLES = [
    "04_fine_tune_qlora.py",
#     "06_fine_tune_qlora_marketplace.py",
]
EXAMPLE_NOTEBOOK_LIST = [
    "01_load_inference.py",
    "02_mlflow_logging_inference.py",
    "03_langchain_inference.py",
#     "08_load_from_marketplace.py",
] + FINE_TUNE_EXAMPLES

@dataclass
class ModelSpecifics:
    model_serving_type_aws: str
    model_serving_type_azure: str
    model_compute_size: str
    model_compute_type_aws: str
    model_compute_type_azure: str
    model_compute_type_gcp: str
    model_fine_tune_type_aws: str
    model_fine_tune_type_azure: str
    model_fine_tune_type_gcp: str


def inference_instance_type(model_size: int, work_type: str) -> str:
    """
    Determines the size category based on the given integer input.

    :param model_size: rounded number of billions of parameters for the model.
    :param work_type: work type ('model_serving', 'inference', 'peft', or 'full_tune').
    :return: A string representing the size category ('small', 'medium', or 'large').
    """
    if model_size < 3:
        if work_type == 'model_serving':
            return {
                "aws": "GPU_SMALL",
                "azure": "GPU_SMALL",
            }
        elif work_type == 'inference':
            return {
                "aws": "`g4dn.xlarge`",
                "azure": "`Standard_NC4as_T4_v3`",
                "gcp": "`g2-standard-4`",
            }
        elif work_type == 'peft':
            return {
                "aws": "`g5.xlarge`",
                "azure": "`Standard_NV36ads_A10_v5`",
                "gcp": "`g2-standard-8` or `a2-highgpu-1g`",
            }
        elif work_type == 'full_tune':
            return {
                "aws": "`g5.xlarge`",
                "azure": "`Standard_NV36ads_A10_v5`",
                "gcp": "`g2-standard-8` or `a2-highgpu-1g`",
            }
    elif model_size < 13:
        if work_type == 'model_serving':
            return {
                "aws": "GPU_MEDIUM",
                "azure": "GPU_LARGE",
            }
        elif work_type == 'inference':
            return {
                "aws": "`g5.xlarge`",
                "azure": "`Standard_NV36ads_A10_v5`",
                "gcp": "`g2-standard-4`",
            }
        elif work_type == 'peft':
            return {
                "aws": "`g5.8xlarge`",
                "azure": "`Standard_NV36ads_A10_v5`",
                "gcp": "`g2-standard-8` or `a2-highgpu-1g`",
            }
        elif work_type == 'full_tune':
            return {
                "aws": "`g5.8xlarge`",
                "azure": "`Standard_NV36ads_A10_v5`",
                "gcp": "`g2-standard-8` or `a2-highgpu-1g`",
            }
    elif 13 <= model_size < 24:
        if work_type == 'model_serving':
            return {
                "aws": "MULTIGPU_MEDIUM",
                "azure": "GPU_LARGE",
            }
        elif work_type == 'inference':
            return {
                "aws": "`g5.12xlarge`",
                "azure": "`Standard_NV36ads_A10_v5`",
                "gcp": "`g2-standard-24`",
            }
        elif work_type == 'peft':
            return {
                "aws": "`g5.8xlarge`",
                "azure": "`Standard_NV36ads_A10_v5`",
                "gcp": "`g2-standard-8` or `a2-highgpu-1g`",
            }
        elif work_type == 'full_tune':
            return {
                "aws": "`g5.48xlarge`",
                "azure": "`Standard_NC48ads_A100_v4`",
                "gcp": "`g2-standard-96` or `a2-highgpu-4g`",
            }
    elif 24 <= model_size < 41:
        if work_type == 'model_serving':
            return {
                "aws": "MULTIGPU_MEDIUM",
                "azure": "GPU_LARGE",
            }
        elif work_type == 'inference':
            return {
                "aws": "`g5.12xlarge`",
                "azure": "`Standard_NC24ads_A100_v4`",
                "gcp": "`g2-standard-48`",
            }
        elif work_type == 'peft':
            return {
                "aws": "`g5.12xlarge`",
                "azure": "`Standard_NC24ads_A100_v4`",
                "gcp": "`a2-ultragpu-1g`",
            }
        elif work_type == 'full_tune':
            return {
                "aws": "`p4d.24xlarge`",
                "azure": "`Standard_NC96ads_A100_v4`",
                "gcp": "`a2-ultragpu-4g`",
            }
    else:
        if work_type == 'model_serving':
            return {
                "aws": "GPU_LARGE_4",
                "azure": "GPU_LARGE_4",
            }
        elif work_type == 'inference':
            return {
                "aws": "`p4d.24xlarge`",
                "azure": "`Standard_NC24ads_A100_v4`",
                "gcp": "`a2-ultragpu-4g`",
            }
        elif work_type == 'peft':
            return {
                "aws": "`p4d.24xlarge`",
                "azure": "`Standard_NC24ads_A100_v4`",
                "gcp": "`a2-ultragpu-4g`",
            }
        elif work_type == 'full_tune':
            return {
                "aws": "`p4d.24xlarge`",
                "azure": "`Standard_NC96ads_A100_v4`",
                "gcp": "`a2-ultragpu-4g`",
            }
        

def get_model_info(hf_model_name: str) -> Dict[str, Any]:
    """
    Gets the latest revision number for a given model name.
    :param hf_model_name: Hugging Face model name.
    :return: The latest revision sha.
    """
    model_info_dict = {}
    hf_model_info = model_info(hf_model_name)
    model_info_dict["revision"] = hf_model_info.sha
    model_info_dict["model_size"] = hf_model_info.safetensors[
                'total'] if hf_model_info.safetensors else None

    return model_info_dict


def should_generate_example(file_name: str, overwrite: bool) -> bool:
    if os.path.exists(file_name) and not overwrite:
        print(f"Skipping {file_name} because it already exists.")
        return False
    return True

def generate_example_notebook(
        model_manifest: Dict[str, Any],
        example_folder: str,
        overwrite: bool=False,
):
    package_loader = TemplateManager.get_filesystem_loader(_SECTIONS_PACKAGE)
    template_manager = TemplateManager(package_loader)
    
    if "prompt_template" in model_manifest:
        prompt_template = model_manifest["prompt_template"]
    else:
        prompt_template = """### Instruction:
{system_prompt}
{instruction}

### Response:\\n"""

    for example in EXAMPLE_NOTEBOOK_LIST:
        file_name = os.path.join(example_folder, example)
        if example in FINE_TUNE_EXAMPLES:
            model_hf_path = f"{model_manifest['hf_org_name']}/{model_manifest['base_model_name']}"
        else:
            model_hf_path = f"{model_manifest['hf_org_name']}/{model_manifest['fine_tuned_model_name']}"
        model_info_dict = get_model_info(model_hf_path)
        model_size = model_info_dict["model_size"] if model_info_dict["model_size"] else model_manifest["model_size"]
        if should_generate_example(file_name, overwrite):
            _logger.info(f"Generating {example}")
            template_manager.dump_template(
                f"text_generation/{example}.jinja",
                file_name,
                model_family_name=model_manifest["model_family_name"],
                model_name=model_manifest["fine_tuned_model_name"],
                base_model_name=model_manifest["base_model_name"],
                fine_tuned_model_name=model_manifest["fine_tuned_model_name"],
                hf_org_name=model_manifest["hf_org_name"],
                compute_type=inference_instance_type(int(model_size), "inference"),
                serving_type=inference_instance_type(int(model_size), "model_serving"),
                peft_type=inference_instance_type(int(model_size), "peft"),
                pip_requirements=model_manifest["pip_requirements"],
                revision=model_info_dict["revision"],
                prompt_template=prompt_template,
                model_size=model_size,
                support_optimized_serving=model_manifest.get("support_optimized_serving", False),
                marketplace_link=model_manifest.get("marketplace_link", ""),
                support_vllm=model_manifest.get("support_vllm", False),
                support_gradient_checkpointing=model_manifest.get("support_gradient_checkpointing", True),
            )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument('--overwrite', default=False, action='store_true')

    args = parser.parse_args()
    model_file = args.model_file
    overwrite = args.overwrite

    if not os.path.exists(model_file):
        raise Exception(f'Model file specified by --model_file {model_file} does not exist.')
    with open(model_file, 'r') as models_file:
        model_manifest = yaml.safe_load(models_file)
    print(model_manifest)

    # Create the folder for the model
    example_folder = f"{model_manifest['model_family_name']}/{model_manifest['model_name']}"
    os.makedirs(example_folder, exist_ok=True)

    generate_example_notebook(model_manifest, example_folder, overwrite)

    

if __name__ == "__main__":
    main()
