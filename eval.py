import argparse
import hydra
import os 

from omegaconf import OmegaConf
from pathlib import Path

from adaflow.workspace.base_workspace import BaseWorkspace


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_inference_steps", 
        default=20,
        type=int,
    )
    parser.add_argument(
        "--eval_exp_dir", 
        default="",
        type=str,
    )
    parser.add_argument(
        "--sampling_method", 
        default=None, 
        type=str,
    )
    parser.add_argument(
        "--evaluate_mode", 
        default="rand_start",   # rand_start or fix_start
        type=str,
    )
    parser.add_argument(
        "--eta", 
        default=None, 
        type=float, 
    )
    parser.add_argument(
        "--training-dataset-path",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--save-rollout-states-path",
        default=None,
        type=str,  # if provided, saves rollout states/actions to this .hdf5 file
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--randomize_color",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--randomize_camera", 
        type=bool,
        default=False
    )
    parser.add_argument(
        "--randomize_dynamics", 
        type=bool,
        default=False
    )
    parser.add_argument(
        "--randomize_lighting", 
        type=bool,
        default=False
    )


    args = parser.parse_args()
    
    if os.path.exists(args.eval_exp_dir): 
        experiment_dir = Path(args.eval_exp_dir) 
    else: 
        raise ValueError("Invalid eval_exp_dir")
    
    cfg = OmegaConf.load(os.path.join(experiment_dir, ".hydra", "config.yaml"))

    if args.training_dataset_path is not None:
        cfg.task.dataset_path = args.training_dataset_path
        cfg.task.env_runner.dataset_path = args.training_dataset_path
        cfg.task.dataset.dataset_path = args.training_dataset_path
    else:
        print(f"No training dataset path provided. Using the default dataset path in ."
              f"hydra/config.yaml: {cfg.task.dataset_path}")


    cfg.task.env_runner.n_train = 0 
    cfg.task.env_runner.n_train_vis = 0
    cfg.task.env_runner.n_test = 100 
    cfg.task.env_runner.n_test_vis = 100
    cfg.task.env_runner.n_envs = 25
    # cfg.task.env_runner.n_test = 20 
    # cfg.task.env_runner.n_test_vis = 20
    # cfg.task.env_runner.n_envs = 20

    if args.debug:
        cfg.task.env_runner.n_envs = 4
        cfg.task.env_runner.n_train = 4 
        cfg.task.env_runner.n_train_vis = 4
        cfg.task.env_runner.n_test = 4
        cfg.task.env_runner.n_test_vis = 4
        cfg.task.env_runner.max_steps = 20

    if args.save_rollout_states_path is not None:
        print(f"Saving rollout states to {args.save_rollout_states_path}")
        cfg.task.env_runner.save_rollout_states_path = args.save_rollout_states_path
    if "evaluate_mode" not in cfg.keys(): 
        cfg.evaluate_mode = args.evaluate_mode
    
    if cfg.name in ["train_adaflow_unet_image"]: 
        cfg.policy.sampling_method = args.sampling_method
        cfg.policy.eta = args.eta
        cfg.policy.num_inference_steps = args.num_inference_steps
        cfg.evaluate_mode = args.evaluate_mode
    
    if cfg.name in ["train_diffusion_unet_image"]: 
        cfg.policy.num_inference_steps = args.num_inference_steps

    cls = hydra.utils.get_class(cfg._target_)

    workspace: BaseWorkspace = cls(cfg, output_dir=experiment_dir)
        
    workspace.eval_only(
        output_dir=experiment_dir,
        randomize_color=args.randomize_color,  
        randomize_camera=args.randomize_camera,
        randomize_lighting=args.randomize_lighting,
        randomize_dynamics=args.randomize_dynamics  
    )

if __name__ == "__main__": 
    main()
