"""
Reference: https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/scripts/dataset_states_to_obs.py

Script to extract observations from low-dimensional simulation states in a robosuite dataset.

Args:
    dataset (str): path to input hdf5 dataset

    output_name (str): name of output hdf5 dataset

    n (int): if provided, stop after n trajectories are processed

    shaped (bool): if flag is set, use dense rewards

    camera_names (str or [str]): camera name(s) to use for image observations. 
        Leave out to not use image observations.

    camera_height (int): height of image observation.

    camera_width (int): width of image observation

    done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a success state.
        If 1, done is 1 at the end of each trajectory. If 2, both.

    copy_rewards (bool): if provided, copy rewards from source file instead of inferring them

    copy_dones (bool): if provided, copy dones from source file instead of inferring them

Example usage:
    
    # extract low-dimensional observations
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name low_dim.hdf5 --done_mode 2
    
    # extract 84x84 image observations
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image.hdf5 \
        --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

    # extract 84x84 image and depth observations
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name depth.hdf5 \
        --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 --depth

    # (space saving option) extract 84x84 image observations with compression and without 
    # extracting next obs (not needed for pure imitation learning algos)
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image.hdf5 \
        --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 \
        --compress --exclude-next-obs

    # use dense rewards, and only annotate the end of trajectories with done signal
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image_dense_done_1.hdf5 \
        --done_mode 1 --dense --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

    # Example with image cropping and resizing:
    python dataset_states_to_obs.py \
        --dataset /path/to/input.hdf5 \
        --output_name output.hdf5 \
        --camera_names agentview \
        --image_crop '{"agentview_depth": [90, 90], "agentview_segmentation_instance": [90, 90]}' \
        --image_resize '{"robot0_eye_in_hand_depth": [90, 90], "robot0_eye_in_hand_segmentation_instance": [90, 90]}'
"""
import os
import json
import h5py
import argparse
import numpy as np
from copy import deepcopy
from tqdm import tqdm

import sys
sys.path.append('/data2/andrew/cpgen-envs')
sys.path.append('/data2/andrew/mimicgen')
sys.path.append('/data2/andrew/adaflow')
sys.path.append('/data2/andrew/robosuite')
sys.path.append('/data2/andrew/robomimic')
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase
import robosuite
from robosuite.wrappers import DomainRandomizationWrapper

from adaflow.real_world.real_inference_util import crop_and_resize_img

import cpgen_envs

def reset_to(self, state, no_return_obs: bool = False):
    """
    Reset to a specific simulator state.

    Args:
        state (dict): current simulator state that contains one or more of:
            - states (np.ndarray): initial state of the mujoco environment
            - model (str): mujoco scene xml
        no_return_obs (bool): if True, do not return observation after setting the simulator state.
            Used to not waste computation when we don't need the observation.
            If False, return observation after setting the simulator state. 
    Returns:
        observation (dict): observation dictionary after setting the simulator state (only
            if "states" is in @state)
    """
    should_ret = False
    if "model" in state:
        self.reset()
        robosuite_version_id = int(robosuite.__version__.split(".")[1])
        if robosuite_version_id <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml
            xml = postprocess_model_xml(state["model"])
        else:
            # v1.4 and above use the class-based edit_model_xml function
            xml = self.env.edit_model_xml(state["model"])
        # save xml to file
        self.env.reset_from_xml_string(xml)                        
        self.env.sim.reset()
        if not self._is_v1:
            # hide teleop visualization after restoring from model
            self.env.sim.model.site_rgba[self.env.eef_site_id] = np.array([0., 0., 0., 0.])
            self.env.sim.model.site_rgba[self.env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
    if "states" in state:
        self.env.sim.set_state_from_flattened(state["states"])
        self.env.sim.forward()
        should_ret = True

    if "goal" in state:
        self.set_goal(**state["goal"])
    if not no_return_obs and should_ret:
        # only return obs if we've done a forward call - otherwise the observations will be garbage
        return self.get_observation()
    return None

# monkey patch robomimic's EnvBase.reset_to with the above function
from robomimic.envs.env_robosuite import EnvRobosuite
EnvRobosuite.reset_to = reset_to

def extract_trajectory(
    env, 
    initial_state, 
    states, 
    actions,
    done_mode,
    camera_names=None, 
    camera_height=84, 
    camera_width=84,
    save_obs: bool = False
):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load to extract information
        actions (np.array): array of actions
        done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a 
            success state. If 1, done is 1 at the end of each trajectory. 
            If 2, do both.
        save_obs (bool): if True, save observations to disk
    """
    assert isinstance(env, EnvBase)
    assert states.shape[0] == actions.shape[0]

    domain_randomize = isinstance(env.env, DomainRandomizationWrapper)

    # load the initial state
    env.reset()
    obs = env.reset_to(initial_state, no_return_obs=True if domain_randomize else False)
    if domain_randomize:
        for modder in env.env.modders:
            modder.update_sim(env.env.sim)
            modder.randomize()
        obs = env.get_observation()

    for obs_key, obs_val in obs.items():
        do_crop = obs_key in args.image_crop
        crop_h, crop_w = args.image_crop[obs_key] if do_crop else (None, None)
        do_resize = obs_key in args.image_resize
        resize_h, resize_w = args.image_resize[obs_key] if do_resize else (None, None)

        if (do_crop or do_resize) and isinstance(obs[obs_key], np.ndarray) and len(obs[obs_key].shape) == 3:
            crop_h = crop_h or obs[obs_key].shape[0]
            crop_w = crop_w or obs[obs_key].shape[1]
            obs[obs_key] = crop_and_resize_img(
                img=obs[obs_key],
                do_crop=do_crop,
                crop_h=crop_h,
                crop_w=crop_w,
                do_resize=do_resize,
                resize_factor_x=resize_w / crop_w if do_resize else 1.0,
                resize_factor_y=resize_h / crop_h if do_resize else 1.0
            )

    # maybe add in intrinsics and extrinsics for all cameras
    camera_info = None
    is_robosuite_env = EnvUtils.is_robosuite_env(env=env)
    if is_robosuite_env:
        camera_info = get_camera_info(
            env=env,
            camera_names=camera_names, 
            camera_height=camera_height, 
            camera_width=camera_width,
        )

    traj = dict(
        obs=[], 
        next_obs=[], 
        rewards=[], 
        dones=[], 
        actions=np.array(actions), 
        states=np.array(states), 
        initial_state_dict=initial_state,
    )
    traj_len = states.shape[0]
    # iteration variable @t is over "next obs" indices
    for t in range(1, traj_len + 1):

        # get next observation
        if t == traj_len:
            # play final action to get next observation for last timestep
            next_obs, _, _, _ = env.step(actions[t - 1])
        else:
            # reset to simulator state to get observation
            next_obs = env.reset_to({"states" : states[t]}, no_return_obs=True if domain_randomize else False)
            if domain_randomize:
                for modder in env.env.modders:
                    modder.randomize()        
                next_obs = env.get_observation()

        # infer reward signal
        # note: our tasks use reward r(s'), reward AFTER transition, so this is
        #       the reward for the current timestep
        r = env.get_reward()

        # infer done signal
        done = False
        if (done_mode == 1) or (done_mode == 2):
            # done = 1 at end of trajectory
            done = done or (t == traj_len)
        if (done_mode == 0) or (done_mode == 2):
            # done = 1 when s' is task success state
            done = done or env.is_success()["task"]
        done = int(done)

        import imageio
        if save_obs:
            def norm_image(img: np.ndarray) -> np.ndarray:
                return img[0] if img.ndim == 4 else img

            def process_images(prefix: str, clamp_max: float) -> None:
                keys = [f"{prefix}_image", f"{prefix}_depth", f"{prefix}_segmentation_instance"]
                imgs = []
                if keys[0] in obs:
                    imgs.append(norm_image(obs[keys[0]].astype(np.uint8)))
                if keys[1] in obs:
                    depth = obs[keys[1]].clip(0, clamp_max)
                    depth = (depth / np.max(depth) * 255).astype(np.uint8)
                    imgs.append(norm_image(np.repeat(norm_image(depth), 3, axis=-1)))
                if keys[2] in obs:
                    seg = (obs[keys[2]] * 255).astype(np.uint8)
                    imgs.append(norm_image(np.repeat(norm_image(seg), 3, axis=-1)))
                if imgs:
                    imageio.imwrite(f'scripts/{prefix}_collage_{t}.png', np.vstack(imgs))
                    print(f'scripts/{prefix}_collage_{t}.png')
            process_images("agentview", 1.4)
            process_images("robot0_eye_in_hand", 0.5)

        # apply image cropping and resizing if specified
        for obs_key, obs_val in obs.items():
            do_crop = obs_key in args.image_crop
            crop_h, crop_w = args.image_crop[obs_key] if do_crop else (None, None)
            do_resize = obs_key in args.image_resize
            resize_h, resize_w = args.image_resize[obs_key] if do_resize else (None, None)

            if (do_crop or do_resize) and isinstance(obs[obs_key], np.ndarray) and len(obs[obs_key].shape) == 3:
                crop_h_next, crop_w_next = args.image_crop[obs_key] if do_crop else next_obs[obs_key].shape[:2]
                crop_h, crop_w = args.image_crop[obs_key] if do_crop else obs[obs_key].shape[:2]

                next_obs[obs_key] = crop_and_resize_img(
                    img=next_obs[obs_key],
                    do_crop=do_crop,
                    crop_h=crop_h_next if do_crop else None,
                    crop_w=crop_w_next if do_crop else None,
                    do_resize=do_resize,
                    resize_factor_x=resize_w / crop_w_next if do_resize else None,
                    resize_factor_y=resize_h / crop_h_next if do_resize else None
                )

        # collect transition
        traj["obs"].append(obs)
        traj["next_obs"].append(next_obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)

        # update for next iter
        obs = deepcopy(next_obs)

    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                try:
                    traj[k][kp] = np.array(traj[k][kp])
                except:
                    import ipdb; ipdb.set_trace()
                    traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj, camera_info


def get_camera_info(
    env,
    camera_names=None, 
    camera_height=84, 
    camera_width=84,
):
    """
    Helper function to get camera intrinsics and extrinsics for cameras being used for observations.
    """

    # TODO: make this function more general than just robosuite environments
    assert EnvUtils.is_robosuite_env(env=env)

    if camera_names is None:
        return None

    camera_info = dict()
    for cam_name in camera_names:
        K = env.get_camera_intrinsic_matrix(camera_name=cam_name, camera_height=camera_height, camera_width=camera_width)
        R = env.get_camera_extrinsic_matrix(camera_name=cam_name) # camera pose in world frame
        if "eye_in_hand" in cam_name:
            # convert extrinsic matrix to be relative to robot eef control frame
            assert cam_name.startswith("robot0")
            eef_site_id = env.base_env.robots[0].eef_site_id['right']
            #eef_pos = np.array(env.base_env.sim.data.site_xpos[env.base_env.sim.model.site_name2id(eef_site_name)])
            eef_pos = np.array(env.base_env.sim.data.site_xpos[eef_site_id])
            #eef_rot = np.array(env.base_env.sim.data.site_xmat[env.base_env.sim.model.site_name2id(eef_site_name)].reshape([3, 3]))
            eef_rot = np.array(env.base_env.sim.data.site_xmat[eef_site_id].reshape([3, 3]))
            eef_pose = np.zeros((4, 4)) # eef pose in world frame
            eef_pose[:3, :3] = eef_rot
            eef_pose[:3, 3] = eef_pos
            eef_pose[3, 3] = 1.0
            eef_pose_inv = np.zeros((4, 4))
            eef_pose_inv[:3, :3] = eef_pose[:3, :3].T
            eef_pose_inv[:3, 3] = -eef_pose_inv[:3, :3].dot(eef_pose[:3, 3])
            eef_pose_inv[3, 3] = 1.0
            R = R.dot(eef_pose_inv) # T_E^W * T_W^C = T_E^C
        camera_info[cam_name] = dict(
            intrinsics=K.tolist(),
            extrinsics=R.tolist(),
        )
    return camera_info

def apply_random_yaw_to_wine_rack_base_xml(xml_str: str) -> str:
    """Applies a random yaw rotation to the wine_glass_rack_base body in the XML string."""
    root = ET.fromstring(xml_str)
    base_body = root.find(".//body[@name='wine_glass_rack_base']")
    if base_body is not None:
        yaw = np.random.uniform(0, 2 * np.pi)
        qw = np.cos(yaw / 2)
        qz = np.sin(yaw / 2)
        base_body.set("quat", f"{qw} 0 0 {qz}")
    return ET.tostring(root, encoding="unicode")

def fix_env_meta(env_meta):
    if env_meta['env_kwargs']['controller_configs'].get("body_parts_controller_configs") is not None:
        env_meta['env_kwargs']['controller_configs']['body_parts'] = \
        env_meta['env_kwargs']['controller_configs']['body_parts_controller_configs']
    env_meta["env_kwargs"]["env_name"] = "SquareRealBetterReal3pvCameraTableAlign"
    env_meta["env_name"] = "SquareRealBetterReal3pvCameraTableAlign"
    return env_meta


DEFAULT_CAMERA_ARGS = {
    "camera_names": None,  # all cameras are randomized
    "randomize_position": True,
    "randomize_rotation": True,
    "randomize_fovy": True,
    "position_perturbation_size": 0.008,
    "rotation_perturbation_size": 0.07,
    "fovy_perturbation_size": 5.0,
}

def dataset_states_to_obs(args):
    if args.depth:
        assert len(args.camera_names) > 0, "must specify camera names if using depth"

    # create environment to use for data processing
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    if args.segmentation:
        env_meta['env_kwargs']['camera_segmentations'] = args.segmentation

    if args.gpu_id != -1:
        env_meta["env_kwargs"]["render_gpu_device_id"] = args.gpu_id
        env_meta["env_kwargs"]["override_gpu"] = True
    else:
        env_meta["env_kwargs"]["override_gpu"] = False


    if env_meta['env_kwargs']['controller_configs'].get("body_parts_controller_configs") is not None:
        env_meta['env_kwargs']['controller_configs']['body_parts'] = env_meta['env_kwargs']['controller_configs']['body_parts_controller_configs']
    
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=args.camera_names, 
        camera_height=args.camera_height, 
        camera_width=args.camera_width, 
        reward_shaping=args.shaped,
        use_depth_obs=args.depth,
    )


    if any([args.randomize_color, args.randomize_camera, args.randomize_dynamics, args.randomize_lighting]):
        env.env = DomainRandomizationWrapper(env.env,
            randomize_color=args.randomize_color,
            randomize_camera=args.randomize_camera,
            randomize_dynamics=args.randomize_dynamics,
            randomize_lighting=args.randomize_lighting,
            randomize_on_reset=True,
            randomize_every_n_steps=1,
            camera_randomization_args=DEFAULT_CAMERA_ARGS
        )

    if args.segmentation:
        import sys
        sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))  # assumes script is in adaflow/scripts/
        # import to monkey patch robosuite envs' get_observation()
        import adaflow.env.robomimic.__init__
        from robomimic.utils.obs_utils import register_obs_key, OBS_MODALITIES_TO_KEYS, OBS_KEYS_TO_MODALITIES
        OBS_MODALITIES_TO_KEYS['segmentation'] = [cam_name + f"_segmentation_{args.segmentation}" for cam_name in args.camera_names]
        OBS_KEYS_TO_MODALITIES.update({v: "segmentation" for v in OBS_MODALITIES_TO_KEYS['segmentation']})

    if args.no_rgb:
        # remove rgb from OBS_MODALITIES_TO_KEYS and OBS_KEYS_TO_MODALITIES
        rgb_keys = OBS_MODALITIES_TO_KEYS.pop("rgb", None)
        for rgb_key in rgb_keys:
            OBS_KEYS_TO_MODALITIES.pop(rgb_key, None)

    print("==== Using environment with the following metadata ====")
    print(json.dumps(env.serialize(), indent=4))
    print("")

    # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(args.dataset, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    # if args.n_traj is not None:
    #     demos = demos[:args.n_traj]

    if args.start is not None:
        demos = demos[args.start:args.end]

    # output file in same directory as input file
    output_path = os.path.join(os.path.dirname(args.dataset), args.output_name)
    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")
    print("input file: {}".format(args.dataset))
    print("output file: {}".format(output_path))

    total_samples = 0
    for ind in tqdm(range(len(demos))):
        ep = demos[ind]
        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
            initial_state["model"] = apply_random_yaw_to_wine_rack_base_xml(initial_state["model"])

        # extract obs, rewards, dones
        actions = f["data/{}/actions".format(ep)][()]
        traj, camera_info = extract_trajectory(
            env=env, 
            initial_state=initial_state, 
            states=states, 
            actions=actions,
            done_mode=args.done_mode,
            camera_names=args.camera_names, 
            camera_height=args.camera_height, 
            camera_width=args.camera_width,
            save_obs=args.save_obs
        )

        # maybe copy reward or done signal from source file
        if args.copy_rewards:
            traj["rewards"] = f["data/{}/rewards".format(ep)][()]
        if args.copy_dones:
            traj["dones"] = f["data/{}/dones".format(ep)][()]

        # store transitions

        # IMPORTANT: keep name of group the same as source file, to make sure that filter keys are
        #            consistent as well
        ep_data_grp = data_grp.create_group(ep)
        norm_act = np.array(traj["actions"])
        traj["actions"] = norm_act
        ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
        ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
        ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
        ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
        for k in traj["obs"]:
            if args.print_mean_std:
                if "agentview_depth" in k:
                    cropped_depth_img = traj['obs'][k][:, 35:-35].clip(0, 1.4)
                    print(f"cropped depth image shape: {cropped_depth_img.shape}")
                    print(f"mean and std of agentview_depth: {np.mean(cropped_depth_img)}, {np.std(cropped_depth_img)}")
                elif "robot0_eye_in_hand_depth" in k:
                    import cv2
                    resized_depth_img = [cv2.resize(traj['obs'][k][i], (90, 90)).clip(0, 0.5) for i in range(traj['obs'][k].shape[0])]
                    resized_depth_img = np.array(resized_depth_img)
                    print(f"resized depth image shape: {resized_depth_img.shape}")
                    print(f"mean and std of robot0_eye_in_hand_depth: {np.mean(resized_depth_img)}, {np.std(resized_depth_img)}")
            if args.compress:
                ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]), compression="gzip")
            else:
                ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
            if not args.exclude_next_obs:
                if args.compress:
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]), compression="gzip")
                else:
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

        # episode metadata
        if is_robosuite_env:
            ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
        ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode

        if camera_info is not None:
            assert is_robosuite_env
            ep_data_grp.attrs["camera_info"] = json.dumps(camera_info, indent=4)

        total_samples += traj["actions"].shape[0]

    # copy over any other attrs from the original hdf5
    for k in f.attrs:
        f_out.attrs[k] = f.attrs[k]

    # copy over all filter keys that exist in the original hdf5
    if "mask" in f:
        f.copy("mask", f_out)

    # global metadata
    data_grp.attrs["total"] = total_samples
    data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
    print("Wrote {} trajectories to {}".format(len(demos), output_path))

    f.close()
    f_out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
    )
    # name of hdf5 to write - it will be in the same directory as @dataset
    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
        help="name of output hdf5 dataset",
    )

    # specify number of demos to process - useful for debugging conversion with a handful
    # of trajectories
    parser.add_argument(
        "--n_traj",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )

    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )

    # flag for printing mean and std of depth images
    parser.add_argument(
        "--print_mean_std",
        action='store_true',
        help="(optional) print mean and std of depth images",
    )

    # flag for reward shaping
    parser.add_argument(
        "--shaped", 
        action='store_true',
        help="(optional) use shaped rewards",
    )

    # camera names to use for observations
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=[],
        help="(optional) camera name(s) to use for image observations. Leave out to not use image observations.",
    )

    parser.add_argument(
        "--camera_height",
        type=int,
        default=84,
        help="(optional) height of image observations",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=84,
        help="(optional) width of image observations",
    )

    # flag for including depth observations per camera
    parser.add_argument(
        "--depth", 
        action='store_true',
        help="(optional) use depth observations for each camera",
    )

    # flag for including segmentation observations per camera
    parser.add_argument(
        "--segmentation",
        type=str,
        default=None,
        help="(optional) use segmentation observations for each camera. Options: instance, element",
    )

    # flag for excluding rgb observations
    parser.add_argument(
        "--no_rgb", 
        action='store_true',
        help="(optional) exclude rgb observations",
    )
    # specifies how the "done" signal is written. If "0", then the "done" signal is 1 wherever 
    # the transition (s, a, s') has s' in a task completion state. If "1", the "done" signal 
    # is one at the end of every trajectory. If "2", the "done" signal is 1 at task completion
    # states for successful trajectories and 1 at the end of all trajectories.
    parser.add_argument(
        "--done_mode",
        type=int,
        default=0,
        help="how to write done signal. If 0, done is 1 whenever s' is a success state.\
            If 1, done is 1 at the end of each trajectory. If 2, both.",
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        default=-1,
    )

    # flag for copying rewards from source file instead of re-writing them
    parser.add_argument(
        "--copy_rewards", 
        action='store_true',
        help="(optional) copy rewards from source file instead of inferring them",
    )

    # flag for copying dones from source file instead of re-writing them
    parser.add_argument(
        "--copy_dones", 
        action='store_true',
        help="(optional) copy dones from source file instead of inferring them",
    )

    # flag to exclude next obs in dataset
    parser.add_argument(
        "--exclude-next-obs", 
        action='store_true',
        help="(optional) exclude next obs in dataset",
    )

    # flag to compress observations with gzip option in hdf5
    parser.add_argument(
        "--compress", 
        action='store_true',
        help="(optional) compress observations with gzip option in hdf5",
    )

    # give image cropping and resizing options: for each camera, specify crop and resize options
    # dictionary mapping camera names to crop and resize operations
    parser.add_argument(
        "--image_crop",
        type=json.loads,
        default={},
        help="(optional) dictionary mapping camera names to crop bounds [top, left, height, width]",
    )
    parser.add_argument(
        "--image_resize", 
        type=json.loads,
        default={},
        help="(optional) dictionary mapping camera names to resize dimensions [height, width]",
    )

    parser.add_argument(
        "--save-obs",
        action='store_true',
        help="(optional) save observations to disk",
    )

    # add flags for each type of DR
    parser.add_argument(
        "--randomize-color", 
        action='store_true',
        help="(optional) enable color domain randomization",
    )

    parser.add_argument(
        "--randomize-camera", 
        action='store_true', 
        help="(optional) enable camera domain randomization",
    )

    parser.add_argument(
        "--randomize-dynamics",
        action='store_true',
        help="(optional) enable dynamics domain randomization", 
    )

    parser.add_argument(
        "--randomize-lighting",
        action='store_true',
        help="(optional) enable lighting domain randomization",
    )

    args = parser.parse_args()
    dataset_states_to_obs(args)
