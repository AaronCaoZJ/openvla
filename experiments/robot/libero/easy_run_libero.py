import os
import sys

LIBERO_ROOT = "/home/zhijun/VLA_Practice/LIBERO"
os.environ["LIBERO_PATH"] = LIBERO_ROOT
sys.path.insert(0, LIBERO_ROOT)
os.chdir(LIBERO_ROOT)

from run_libero_eval import GenerateConfig, eval_libero

cfg = GenerateConfig(
    model_family="openvla",
    pretrained_checkpoint="/home/zhijun/.cache/huggingface/hub/models--openvla--openvla-7b-finetuned-libero-spatial",
    task_suite_name="libero_spatial",
    center_crop=True,
    # 其他参数可按需填写
)

# print(sys.path)

eval_libero(cfg)