import numpy as np

import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.obs_utils import process_frame, unprocess_frame, Modality


def get_observation(self, di=None, vis: bool = False):
    """
    Get current environment observation dictionary.

    Args:
        di (dict): current raw observation dictionary from robosuite to wrap and provide 
            as a dictionary. If not provided, will be queried from robosuite.
    """
    if di is None:
        di = self.env._get_observations(force_update=True) if self._is_v1 else self.env._get_observation()

    ret = {}
    for k in di:
        if (k in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(key=k, obs_modality="rgb"):
            # by default images from mujoco are flipped in height
            ret[k] = di[k][::-1]
            if self.postprocess_visual_obs:
                ret[k] = ObsUtils.process_obs(obs=ret[k], obs_key=k)
        elif (k in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(key=k, obs_modality="depth"):
            # by default depth images from mujoco are flipped in height
            ret[k] = di[k][::-1]
            if len(ret[k].shape) == 2:
                ret[k] = ret[k][..., None] # (H, W, 1)
            assert len(ret[k].shape) == 3
            # scale entries in depth map to correspond to real distance.
            ret[k] = self.get_real_depth_map(ret[k])
            if self.postprocess_visual_obs:
                ret[k] = ObsUtils.process_obs(obs=ret[k], obs_key=k)
        ##################################
        # Start: Add binary segmentation #
        elif (k in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(key=k, obs_modality="segmentation"):
            di[k][di[k] >= 1] = 1  # manually separate into table vs non-table
            ret[k] = di[k][::-1] #.transpose(2, 0, 1)  # H X W X C to C X H X W
            if self.postprocess_visual_obs:
                ret[k] = ObsUtils.process_obs(obs=ret[k], obs_key=k)

            if vis:
                import imageio.v3 as iio
                iio.imwrite("binary_segmentation.png", (ret[k].squeeze() * 255).astype(np.uint8))
                iio.imwrite("depth.png", (ret["agentview_depth"].squeeze() * 255).astype(np.uint8))
                iio.imwrite("robot0_eye_in_hand.png", (ret["robot0_eye_in_hand_depth"].squeeze() * 255).astype(np.uint8))
                import ipdb;ipdb.set_trace()
        # End: Add binary segmentation #
        ################################
    # "object" key contains object information
    ret["object"] = np.array(di["object-state"])

    if self._is_v1:
        for robot in self.env.robots:
            # add all robot-arm-specific observations. Note the (k not in ret) check
            # ensures that we don't accidentally add robot wrist images a second time
            pf = robot.robot_model.naming_prefix
            for k in di:
                if k.startswith(pf) and (k not in ret) and \
                        (not k.endswith("proprio-state")):
                    ret[k] = np.array(di[k])
    else:
        # minimal proprioception for older versions of robosuite
        ret["proprio"] = np.array(di["robot-state"])
        ret["eef_pos"] = np.array(di["eef_pos"])
        ret["eef_quat"] = np.array(di["eef_quat"])
        ret["gripper_qpos"] = np.array(di["gripper_qpos"])
    return ret

from robomimic.envs.env_robosuite import EnvRobosuite

# monkey patch the function
EnvRobosuite.get_observation = get_observation
print("Patched get_observation() in robomimic's robosuite_envs")



class SegmentationModality(Modality):
    """
    Modality for depth observations
    """
    name = "segmentation"

    @classmethod
    def _default_obs_processor(cls, obs):
        """
        Given segmentation fetched from dataset, process for network input. Converts array
        to float (from uint8), normalizes pixels from range [0, 1] to [0, 1], and channel swaps
        from (H, W, C) to (C, H, W).

        Args:
            obs (np.array or torch.Tensor): segmentation array

        Returns:
            processed_obs (np.array or torch.Tensor): processed segmentation
        """
        return process_frame(frame=obs, channel_dim=1, scale=1.)

    @classmethod
    def _default_obs_unprocessor(cls, obs):
        """
        Given segmentation prepared for network input, prepare for saving to dataset.
        Inverse of @process_segmentation.

        Args:
            obs (np.array or torch.Tensor): segmentation array

        Returns:
            unprocessed_obs (np.array or torch.Tensor): segmentation passed through
                inverse operation of @process_segmentation
        """
        return unprocess_frame(frame=obs, channel_dim=1, scale=1.)

# monkey patch the class
ObsUtils.SegmentationModality = SegmentationModality
print("Patched SegmentationModality in robomimic's obs_utils")
