# 🕶️ Get Started

## Setup OpenVLA
```bash
# Create and activate conda environment
conda create -n openvla python=3.10 -y
conda activate openvla

# Clone and install this openvla repo
git clone https://github.com/AaronCaoZJ/openvla.git
cd openvla
pip install -e .
```

## Setup LIBERO Simulation Benchmark

```bash
# Clone and install this LIBERO repo
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

Additionally, install other required packages:

```bash
cd openvla
# Remember to modify libero_requirements.txt in advance and specify *numpy==1.26.4* to avoid package version conflicts
pip install -r experiments/robot/libero/libero_requirements.txt

# Install PyTorch. Below is a sample command to do this on *RTX5090*, but you should check the following link
pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.8.3" --no-build-isolation
```

It may be necessary to create a symbolic link:
```bash
cd experiments/robot/libero
mkdir -p libero
cd libero

ln -s /home/zhijun/VLA_Practice/LIBERO/libero/libero/init_files init_files
ln -s /home/zhijun/VLA_Practice/LIBERO/libero/libero/assets assets
ln -s /home/zhijun/VLA_Practice/LIBERO/libero/libero/bddl_files bddl_files
```

<br>

# 🎯 Released Pretrained Checkpoints

The OpenVLA official team releases two OpenVLA models trained as part of the paper, with checkpoints, configs, and model cards available [on the
HuggingFace page](https://huggingface.co/openvla):
- [`openvla-7b`](https://huggingface.co/openvla/openvla-7b): The flagship model from the paper, trained from 
  the Prismatic `prism-dinosiglip-224px` VLM (based on a fused DINOv2 and SigLIP vision backbone, and Llama-2 LLM). 
  Trained on a large mixture of datasets from Open X-Embodiment spanning 970K trajectories 
  ([mixture details - see "Open-X Magic Soup++"](./prismatic/vla/datasets/rlds/oxe/mixtures.py)).
- [`openvla-v01-7b`](https://huggingface.co/openvla/openvla-7b-v01): An early model used during development, trained from
  the Prismatic `siglip-224px` VLM (singular SigLIP vision backbone, and a Vicuña v1.5 LLM). Trained on the same mixture
  of datasets as [Octo](https://github.com/octo-models/octo), but for significantly fewer GPU hours than the final model 
  ([mixture details - see "Open-X Magic Soup"](./prismatic/vla/datasets/rlds/oxe/mixtures.py)).

To download, use the `huggingface_hub`:
```python
from huggingface_hub import snapshot_download

snapshot_download(
  repo_id="openvla/desired/model/checkpoint",
  repo_type="model",
  local_dir="/desired/models/storage/path"
)
```

<br>

# 🔎 Evaluating with LIBERO

In the [updated OpenVLA paper (v2)](https://arxiv.org/abs/2406.09246), the authors discuss fine-tuning OpenVLA
on a simulated benchmark, [LIBERO](https://libero-project.github.io/main.html), in Appendix E.
Please see the paper for details, such as how the authors modify the provided demonstration datasets to
improve the overall performance of all methods.

The OpenVLA team fine-tuned OpenVLA via LoRA (r=32) on four LIBERO task suites independently: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-10 (also called LIBERO-Long).
The four checkpoints are available on Hugging Face:

* [openvla/openvla-7b-finetuned-libero-spatial](https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial)
* [openvla/openvla-7b-finetuned-libero-object](https://huggingface.co/openvla/openvla-7b-finetuned-libero-object)
* [openvla/openvla-7b-finetuned-libero-goal](https://huggingface.co/openvla/openvla-7b-finetuned-libero-goal)
* [openvla/openvla-7b-finetuned-libero-10](https://huggingface.co/openvla/openvla-7b-finetuned-libero-10)

To download, use the `huggingface_hub`.

## Launching LIBERO Evaluations

To start evaluation with one of these checkpoints, run one of the commands below. Each will automatically download the appropriate checkpoint listed above.

```bash
# Before run eval, set "init_states = torch.load(init_states_path, *weights_only=False*)" in home/zhijun/VLA_Practice/LIBERO/libero/libero/benchmark/__init__.py

# Launch LIBERO-Spatial evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True

# Launch LIBERO-Object evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --center_crop True

# Launch LIBERO-Goal evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --center_crop True

# Launch LIBERO-10 (LIBERO-Long) evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True
```

* The evaluation script will run 500 trials by default (10 tasks x 50 episodes each). You can modify the number of
  trials per task by setting `--num_trials_per_task`. You can also change the random seed via `--seed`.
* **NOTE: Setting `--center_crop True` is important** because the OpenVLA team fine-tuned OpenVLA with random crop augmentations
  (the team took a random crop with 90% area in every training sample, so at test time the team simply takes the center 90% crop).
* The evaluation script logs results locally. You can also log results in Weights & Biases
  by setting `--use_wandb True` and specifying `--wandb_project <PROJECT>` and `--wandb_entity <ENTITY>`.

## If Need the Modified Versions of LIBERO Datasets

To download the modified versions of the LIBERO datasets that the OpenVLA team used in the fine-tuning
experiments, run the command below. This will download the LIBERO-Spatial, LIBERO-Object, LIBERO-Goal,
and LIBERO-10 datasets in RLDS data format (~10 GB total). You can use these to fine-tune OpenVLA or
train other methods. This step is optional since the OpenVLA team provides pretrained OpenVLA checkpoints below.
(Also, you can find the script the team used to generate the modified datasets in raw HDF5 format
[here](experiments/robot/libero/regenerate_libero_dataset.py) and the code used to convert these
datasets to the RLDS format [here](https://github.com/moojink/rlds_dataset_builder).)
```bash
git clone git@hf.co:datasets/openvla/modified_libero_rlds
```

## OpenVLA Fine-Tuning Results

| Method | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-Long | Average |
|--------|----------------|---------------|-------------|-------------|---------|
| Diffusion Policy from scratch | 78.3 ± 1.1% | **92.5 ± 0.7%** | 68.3 ± 1.2% | 50.5 ± 1.3% | 72.4 ± 0.7% |
| Octo fine-tuned | 78.9 ± 1.0% | 85.7 ± 0.9% | **84.6 ± 0.9%** | 51.1 ± 1.3% | 75.1 ± 0.6% |
| OpenVLA fine-tuned (the paper) | **84.7 ± 0.9%** | 88.4 ± 0.8% | 79.2 ± 1.0% | **53.7 ± 1.3%** | **76.5 ± 0.6%** |

Each success rate is the average over 3 random seeds x 500 rollouts each (10 tasks x 50 rollouts per task).

<br>

# 🌵 Repository Structure

High-level overview of repository/project file-tree:

+ `prismatic` - Package source; provides core utilities for model loading, training, data preprocessing, etc.
+ `vla-scripts/` - Core scripts for training, fine-tuning, and deploying VLAs.
+ `experiments/` - Code for evaluating OpenVLA policies in robot environments.
+ `LICENSE` - All code is made available under the MIT License; happy hacking!
+ `Makefile` - Top-level Makefile (by default, supports linting - checking & auto-fix); extend as needed.
+ `pyproject.toml` - Full project configuration details (including dependencies), as well as tool configurations.
+ `README.md` - You are here!