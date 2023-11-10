import argparse
from dataclasses import dataclass
import os
import yaml

from template import TemplateManager

_SECTIONS_PACKAGE = "./template/templates"

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
    "01_load_inference_vllm.py",
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

    elif 13 <= model_size < 40:
        return ModelSpecifics(
            model_serving_type_aws="MULTIGPU_MEDIUM",
            model_serving_type_azure="GPU_LARGE",
            model_compute_size="4xA10 or 1xA100",
            model_compute_type_aws="`g5.12xlarge`",
            model_compute_type_azure="`Standard_NV72ads_A10_v5` or `Standard_NC24ads_A100_v4`",
            model_compute_type_gcp="`g2-standard-48` or `a2-highgpu-1g`",
            model_fine_tune_type_aws="`g5.8xlarge`",
            model_fine_tune_type_azure="`Standard_NV36ads_A10_v5`",
            model_fine_tune_type_gcp="`g2-standard-8` or `a2-highgpu-1g`"
        )
    else:
        return ModelSpecifics(
            model_serving_type_aws="GPU_LARGE",
            model_serving_type_azure="GPU_LARGE",
            model_compute_size="8xA10 or 4xA100",
            model_compute_type_aws="`p4d.24xlarge` or `g5.48xlarge`",
            model_compute_type_azure="`Standard_NC48ads_A100_v4`",
            model_compute_type_gcp="`a2-highgpu-8g` or  `g2-standard-96`",
            model_fine_tune_type_aws="`p4d.24xlarge`",
            model_fine_tune_type_azure="`Standard_NC48ads_A100_v4`",
            model_fine_tune_type_gcp="`a2-highgpu-8g`"
        )


for example in example_list:
    file_name = os.path.join(example_folder, example)
    if os.path.exists(file_name) and not overwrite:
        print(f"Skipping {file_name} because it already exists.")
        continue

    template_manager.dump_template(
        f"{example}.jinja",
        file_name,
        model_name=model_manifest["instruct_model_name"],
        base_model_name=model_manifest["base_model_name"],
        hf_org_name=model_manifest["hf_org_name"],

    )
