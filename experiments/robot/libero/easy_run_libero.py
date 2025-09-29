import sys
sys.path = ["/home/zhijun/VLA_Practice/LIBERO"] + sys.path  # 保证优先加载正确的 LIBERO

import os
os.chdir("/home/zhijun/VLA_Practice/LIBERO")

from run_libero_eval import GenerateConfig, eval_libero

cfg = GenerateConfig(
    model_family="openvla",
    pretrained_checkpoint="/home/zhijun/.cache/huggingface/hub/models--openvla--openvla-7b-finetuned-libero-spatial",
    task_suite_name="libero_spatial",
    center_crop=True,
    # 其他参数可按需填写
)

print(sys.path)

eval_libero(cfg)