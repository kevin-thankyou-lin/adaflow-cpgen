# Introduction

Policy training repo used in [Constraint-Preserving Data Generation
for Visuomotor Policy Generalization](cp-gen.github.io), from [AdaFlow: Imitation Learning with Variance-Adaptive Flow-Based Policies](https://arxiv.org/abs/2402.04292).


**0.** CUDA on workstation

To use cuda, add the following to `~/.bashrc`:

```bash
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
```

**1.** Environment Setup

If not installed, install [mamba](https://github.com/conda-forge/miniforge?tab=readme-ov-file#unix-like-platforms-macos--linux).

Set up your Python environment by running the following commands:
```bash
mamba create -n adaflow python=3.10
mamba activate adaflow
mamba install gym==0.21.0
mamba install pytorch3d==0.7.7
```

Next, install the pytorch version that matches the cuda version on your machine:

E.g., if `nvcc --version` gives `12.1`, use: 

`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

Finally,

```bash
pip install -r requirements.txt
```

**2.** Policy training

Download training datasets. Refer to the instructions [here](https://github.com/kevin-thankyou-lin/cpgen?tab=readme-ov-file#download-generated-datasets).

Head over to a relevant task file (in this case, `adaflow/config/task/square_image_abs.yaml`), and:

1) update the `dataset_path` to
the desired dataset e.g. `dataset_path: &dataset_path datasets/generated/ThreePieceAssemblyWide/2025-03-24-05-33-45/E73/E73-original-action-noisy-state-action-std-0.03-rgb-84-84.hdf5`
2) update the `task_name` to the relevant task name e.g. `ThreePieceAssemblyWide`.

Then, run:

```bash
HYDRA_FULL_ERROR=1 python train.py --config-name=train_diffusion_unet_ddpm_image_workspace_robomimic task=square_image_abs task.dataset_type=ph
```

**3.** Policy evaluation


Locate or download a policy checkpoint folder to evaluate (e.g. `https://huggingface.co/cpgen/cpgen-policies/tree/main/policy_checkpoints/ThreePieceAssemblyWide/2025-03-24-05-33-45_E73/checkpoints`).

To download:

```bash
git lfs install
git clone https://huggingface.co/datasets/cpgen/cpgen-policies
# fetch actual data
git lfs pull <path/to/file>
```

If downloading from hugging face, also need to ensure dataset is in the correct path. Download dataset from: `https://huggingface.co/datasets/cpgen/datasets/tree/main/datasets/generated`. Ensure the `dataset_path` values in `.hydra/config.yaml` (e.g. `policy_checkpoints/ThreePieceAssemblyWide/2025-03-24-05-33-45_E73/.hydra/config.yaml`) to the location where an actual dataset is stored.

Then, run:

```bash
HYDRA_FULL_ERROR=1 PYTHONPATH=.:../robosuite python eval.py  --eval_exp_dir <path/to/exp/dir> --num_inference_steps 100 
```

Example of `<path/to/exp/dir>` is `<path/to>/policy_checkpoints/NutAssemblySquare/v0`


## Citation

If you find this implementation helpful, please consider citing the original work as follows:
```
@article{hu2024adaflow,
  title={AdaFlow: Imitation Learning with Variance-Adaptive Flow-Based Policies},
  author={Hu, Xixi and Liu, Bo and Liu, Xingchao and Liu, Qiang},
  journal={arXiv preprint arXiv:2402.04292},
  year={2024}
}

@article{liu2022flow,
  title={Flow straight and fast: Learning to generate and transfer data with rectified flow},
  author={Liu, Xingchao and Gong, Chengyue and Liu, Qiang},
  journal={arXiv preprint arXiv:2209.03003},
  year={2022}
}
```

## Thanks
A Large portion of this codebase is built upon [Diffusion Policy](https://github.com/real-stanford/diffusion_policy).






