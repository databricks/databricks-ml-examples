import argparse
from dataclasses import dataclass
from huggingface_hub import model_info
import os
import yaml
import logging
import sys

from template import TemplateManager

from prompt_template import prompt_template_dict, prompt_template_end_dict

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
_logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
_logger.addHandler(handler)

_SECTIONS_PACKAGE = "./scripts/templates/"

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

package_loader = TemplateManager.get_filesystem_loader(_SECTIONS_PACKAGE)
template_manager = TemplateManager(package_loader)

# Create the folder for the model
example_folder = f"{model_manifest['model_family_name']}/{model_manifest['model_name']}"
os.makedirs(example_folder, exist_ok=True)

example_list = [
    "01_load_inference.py",
#    "01_load_inference_vllm.py",
    "02_mlflow_logging_inference.py",
    "03_serve_driver_proxy.py",
    "06_fine_tune_qlora.py",
    "06_fine_tune_qlora_marketplace.py",
    "08_load_from_marketplace.py",
]


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

    :param model_size: model size.
    :param work_type: work type ('model_serving', 'inference', 'peft', or 'full_tune').
    :return: A string representing the size category ('small', 'medium', or 'large').
    """
    if model_size < 13:
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


def get_latest_revision(hf_model_name: str) -> str:
    """
    Gets the latest revision number for a given model name.
    :param hf_model_name: Hugging Face model name.
    :return: The latest revision sha.
    """
    hf_model_info = model_info(hf_model_name)
    return hf_model_info.sha


def should_generate_example(example: str) -> bool:
    file_name = os.path.join(example_folder, example)
    if os.path.exists(file_name) and not overwrite:
        print(f"Skipping {file_name} because it already exists.")
        return False
    return True


def generate_example_notebook():
    revision = get_latest_revision(f"{model_manifest['hf_org_name']}/{model_manifest['instruct_model_name']}")

    for example in example_list:
        if should_generate_example(example):
            _logger.info(f"Generating {example}")
            file_name = os.path.join(example_folder, example)
            template_manager.dump_template(
                f"base_model/{example}.jinja",
                file_name,
                model_family_name=model_manifest["model_family_name"],
                model_name=model_manifest["instruct_model_name"],
                base_model_name=model_manifest["base_model_name"],
                instruct_model_name=model_manifest["instruct_model_name"],
                hf_org_name=model_manifest["hf_org_name"],
                compute_type=inference_instance_type(int(model_manifest["model_size"]), "inference"),
                serving_type=inference_instance_type(int(model_manifest["model_size"]), "model_serving"),
                peft_type=inference_instance_type(int(model_manifest["model_size"]), "peft"),
                pip_requirements=model_manifest["pip_requirements"],
                revision=revision,
                prompt_template=prompt_template_dict[model_manifest.get("prompt_template", "NousResearch")],
                prompt_end=prompt_template_end_dict[model_manifest.get("prompt_template", "NousResearch")],
                model_size=model_manifest["model_size"],
                marketplace_link=model_manifest.get("marketplace_link", ""),
                support_vllm=model_manifest.get("support_vllm", False),
            )

if __name__ == "__main__":
    generate_example_notebook()