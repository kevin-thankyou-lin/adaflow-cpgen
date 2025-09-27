import torch
import dill
import hydra
from adaflow.workspace.base_workspace import BaseWorkspace
from adaflow.dataset.base_dataset import LinearNormalizer
import re


NORMALIZER_PREFIX_LENGTH = 11
MODEL_PREFIX_LENGTH = 6


def state_dict_to_model(state_dict, pattern=r'model\.'):
    new_state_dict = {}
    prefix = re.compile(pattern)

    for k, v in state_dict["state_dicts"]["model"].items():
        if re.match(prefix, k):
            # Remove prefix
            new_k = k[MODEL_PREFIX_LENGTH:]
            new_state_dict[new_k] = v

    return new_state_dict

def load_normalizer(workspace_state_dict):
    keys = workspace_state_dict['state_dicts']['model'].keys()
    normalizer_keys = [key for key in keys if 'normalizer' in key]
    normalizer_dict = {key[NORMALIZER_PREFIX_LENGTH:]: workspace_state_dict['state_dicts']['model'][key] for key in normalizer_keys}

    normalizer = LinearNormalizer()
    normalizer.load_state_dict(normalizer_dict)

    return normalizer

def get_policy(ckpt_path, cfg = None, dataset_path = None, consistency_policy: bool = False):
    """
    Returns loaded policy from checkpoint
    If cfg is None, the ckpt's saved cfg will be used
    """
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill, map_location='cuda:0')
    cfg = payload['cfg'] if cfg is None else cfg

    if consistency_policy:
        cfg.training.inference_mode = True
        cfg.training.online_rollouts = False

    if dataset_path is not None:
        cfg.task.dataset.dataset_path = dataset_path
        cfg.task.envrunner.dataset_path = dataset_path

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_checkpoint(path=ckpt_path, exclude_keys=['optimizer'])
    workspace_state_dict = torch.load(ckpt_path, map_location='cuda:0', weights_only=False) #   # allow loading non-tensor globals (omegaconf, etc.))
    normalizer = load_normalizer(workspace_state_dict)

    if cfg.training.use_ema:
        print(f"Using ema model from {ckpt_path}")
        policy = workspace.ema_model
    else:
        policy = workspace.model

    policy.set_normalizer(normalizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device).eval()

    print(f"Loaded policy from {ckpt_path}")

    return policy

def get_cfg(ckpt_path):
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    return cfg


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=str)
    args = parser.parse_args()

    policy = get_policy(args.ckpt_path)
    print(policy)
