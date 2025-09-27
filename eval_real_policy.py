"""Use meta policy mechanism for running the policy."""

from collections import defaultdict
from torch.distributions import MixtureSameFamily
import os
import argparse
import h5py
import random
import ast
import torch
import pickle

# PYTHONPATH=sam2:sam2/sam2 python eval_real_policy.py

# from deoxys_control.deoxys.deoxys.franka_interface import FrankaInterface
# from deoxys.k4a_interface import K4aInterface
# from rpl_vision_utils.networking.camera_redis_interface import CameraRedisSubInterface
from deoxys_vision.networking.camera_redis_interface import CameraRedisPubInterface
from deoxys.franka_interface.franka_interface import (
    FrankaInterface,
)
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.input_utils import input2action
from deoxys_vision.networking.camera_redis_interface import CameraRedisPubInterface
from deoxys_vision.camera.rs_interface import RSInterface
from deoxys_vision.utils.img_utils import preprocess_color, preprocess_depth, save_depth
from deoxys_vision.utils.camera_utils import (
    assert_camera_ref_convention,
    get_camera_info,
)
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


from deoxys.utils import YamlConfig
import torch

from easydict import EasyDict

import numpy as np
import cv2
import imageio
from easydict import EasyDict
import json
import time

import hydra
from omegaconf import OmegaConf, DictConfig
import yaml
from easydict import EasyDict
from hydra.experimental import compose, initialize
import pprint
from pathlib import Path
from typing import List, Dict, Any
from adaflow.utils import get_policy
from adaflow.model.common.rotation_transformer import RotationTransformer
import matplotlib
import matplotlib.pyplot as plt
from torchvision import transforms
from hydra.core.global_hydra import GlobalHydra

from cpgen_envs.environments.manipulation.nut_assembly import (
    SquareRealBetterReal3pvCameraTableAlign,
)

from deoxys.utils.ik_utils import IKWrapper
import pyrealsense2 as rs

# from deoxys.utils.io_devices import SpaceMouse
# from deoxys.utils.input_utils import input2action
from deoxys import config_root
import imageio

# from rpl_vision_utils.utils import img_utils as ImgUtils

from deoxys_vision.utils import img_utils as ImgUtils


matplotlib.use("agg")


DEMO = -1
ROLLOUT = 0
INTV = 1
use_depth = True

from queue import Queue, Empty
from threading import Thread, Event
from typing import Optional
import cv2

# GlobalHydra.instance().clear()


def mat2quat(rmat):
    """
    Converts given rotation matrix to quaternion.

    Args:
        rmat (np.array): 3x3 rotation matrix

    Returns:
        np.array: (x,y,z,w) float quaternion angles
    """
    M = np.asarray(rmat).astype(np.float32)[:3, :3]

    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]
    # symmetric matrix K
    K = np.array(
        [
            [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
            [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
            [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    )
    K /= 3.0
    # quaternion is Eigen vector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    inds = np.array([3, 0, 1, 2])
    q1 = V[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])
    return q1[inds]


class ImageRenderer:
    def __init__(self) -> None:
        self.frame_queue: Queue = Queue()
        self.stop_event: Event = Event()

    def start(self) -> None:
        self.thread = Thread(target=self._render_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join()

    def push_frame(self, frame) -> None:
        # Enqueues new frames from the camera stream
        self.frame_queue.put(frame)

    def _render_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
                cv2.imshow("Camera Stream", frame)
                cv2.waitKey(1)
            except Empty:
                pass


def execute_ik_result(
    robot_interface, controller_type, controller_cfg, joint_traj, gripper_action
):
    valid_input = False
    while not valid_input:
        try:
            execute = input(f"Excute or not? (enter 0 - No or 1 - Yes)")
            execute = bool(int(execute))
            valid_input = True
        except ValueError:
            print("Please input 1 or 0!")
            continue

    if execute:
        for joint in joint_traj:
            # This example assumes the gripper is open
            print("real gripper action is ", gripper_action)
            action = joint.tolist() + [gripper_action]
            robot_interface.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )
    # robot_interface.close()
    return bool(execute)


def segment_black_table(image: np.ndarray, upper_black=[180, 180, 180]) -> np.ndarray:
    """Assumes image is the original image from the camera (H, W, C, uint8).
    
    Returns a binary mask of shape (H, W, 1), where black regions are 0 and valid regions are 1."""
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for black color segmentation (exclude black)
    lower_black = np.array([0, 0, 0])  # Black has very low saturation and value
    # upper_black = np.array([175, 175, 170])  # Limit saturation and value to avoid black pixels

    # Create the mask to segment the black areas (black pixels will be white, others black)
    mask = cv2.inRange(hsv, lower_black, np.array(upper_black))

    # Invert the mask to keep everything except black
    inverted_mask = cv2.bitwise_not(mask)[None, ...]

    inverted_mask[np.where(inverted_mask == 255)] = 1

    assert inverted_mask.max() <= 1, "segmentation output should be between zero and one"
    return inverted_mask
    # Transpose the mask back to (C, H, W) to match the input image format (channel-first format)
    return np.transpose(inverted_mask, (2, 0, 1))

def segment_black_table_sam(
    image: np.ndarray,
    model_checkpoint: str = "../sam2/checkpoints/sam2.1_hiera_large.pt",
    model_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    device: str = "cuda",
    save_dir: Path = Path("sam2_masks"),
    black_v_threshold: int = 80,
    save_image_and_mask: bool = True,
    save_individual_masks: bool = True,
    resize: bool = True,
    load_test: bool = False,
) -> np.ndarray:
    """
    Segments the input image (H, W, C, uint8) to separate non-black regions using SAM2.
    Returns a binary mask of shape (1, H, W) where valid regions are marked as 1.
    """
    image_np = image.copy() if image.dtype == np.uint8 else (image * 255).astype(np.uint8)

    if load_test:
        # Load custom image for custom segmentation
        image_np = cv2.imread("adaflow/img.png")
        if image_np is None:
            raise RuntimeError("Failed to load image")
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    if resize:
        image_np = cv2.resize(image_np, (320 // 2, 180 // 2), interpolation=cv2.INTER_AREA)   

    save_dir.mkdir(parents=True, exist_ok=True)
    sam_model = build_sam2(model_config, model_checkpoint).to(device)
    mask_generator = SAM2AutomaticMaskGenerator(sam_model, stability_score_thresh=0.80, min_mask_region_area=40)  
    # allow more masks since can threshold output
    sam_masks = mask_generator.generate(image_np)

    hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    hsv_tensor = torch.tensor(hsv_image, dtype=torch.float32, device=device)

    H, W, _ = image_np.shape
    final_mask = torch.zeros((H, W), dtype=torch.uint8, device=device)

    for i, mask in enumerate(sam_masks):
        seg_np = mask["segmentation"]  # (H, W)
        seg_tensor = torch.tensor(seg_np, dtype=torch.uint8, device=device)
        segment_pixels = hsv_tensor[seg_tensor > 0]
        if segment_pixels.numel() == 0:
            continue
        avg_hsv = segment_pixels.float().mean(dim=0)
        # keep segment if value (brightness) is not too low
        if avg_hsv[2] > black_v_threshold:
            final_mask[seg_tensor > 0] = 1

        if save_individual_masks:
            print(f"Segment pixels shape: {segment_pixels.shape}")
            print(f"Average HSV for segment: {avg_hsv}")
            mask_img = (mask["segmentation"] * 255).astype(np.uint8)
            # save seg_np to disk
            imageio.imwrite("adaflow/seg.png", mask_img)
            mask_path = save_dir / f"mask_{i}.png"
            imageio.imwrite(mask_path, mask_img)
            # import ipdb; ipdb.set_trace()
            print(f"Saved mask {i} to {mask_path}")

    if save_image_and_mask:
        # Visualize original image and final mask side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(image_np)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        ax2.imshow(final_mask.cpu().numpy(), cmap='gray')
        ax2.set_title('Final Mask')
        ax2.axis('off')
        
        plt.savefig(save_dir / 'image_and_mask.png')
        plt.close()
        print(f"Saved image and mask to {save_dir / 'image_and_mask.png'}")

    return final_mask.unsqueeze(0).cpu().numpy()  # Shape: (1, H, W)


def undo_transform_action(action, rotation_transformer):
    raw_shape = action.shape
    if raw_shape[-1] == 20:
        # dual arm
        action = action.reshape(-1, 2, 10)

    d_rot = action.shape[-1] - 4
    pos = action[..., :3]
    rot = action[..., 3 : 3 + d_rot]
    gripper = action[..., [-1]]
    rot = rotation_transformer.inverse(rot)
    uaction = np.concatenate([pos, rot, gripper], axis=-1)

    if raw_shape[-1] == 20:
        # dual arm
        uaction = uaction.reshape(*raw_shape[:-1], 14)

    return uaction


# final_obs[key][0][0] = last_obs[key][0][0]
# RuntimeError: The expanded size of the tensor (90) must match the existing size (160) at non-singleton dimension 1.  Target sizes: [1, 90, 160].  Tensor sizes: [90, 160, 1]
def upd_obs(last_obs, cur_obs, final_obs):
    if last_obs == None:
        last_obs = cur_obs

    for key in final_obs.keys():
        print(key)
        final_obs[key][0][0] = cur_obs[key][0][0]
        final_obs[key][0][1] = last_obs[key][0][0]

    last_obs = cur_obs


def setup_camera_interface(
    camera_ref="rs_0",
    host="172.16.0.1",
    port=6379,
    redis_password="iloverobotsyifeng",
    use_rgb=True,
    use_depth=True,
    use_rec=False,
    rgb_convention="rgb",
    serial=-1,
):
    """
    Set up camera interface based on provided arguments.

    Args:
        camera_ref (str): Camera reference.
        host (str): Host for Redis connection.
        port (int): Port for Redis connection.
        use_rgb (bool): Whether to use RGB images.
        use_depth (bool): Whether to use depth images.
        use_rec (bool): Whether to rectify images.
        rgb_convention (str): RGB convention to use.

    Returns:
        camera_interface: Instantiated camera interface object.
        camera2redis_pub_interface: Instantiated CameraRedisPubInterface object.
        node_config: EasyDict containing node configuration options.
    """
    assert_camera_ref_convention(camera_ref)
    camera_info = get_camera_info(camera_ref)

    # Print camera information
    print(
        f"This node runs with the camera {camera_info.camera_type} with id {camera_info.camera_id}"
    )

    # Create camera configuration
    camera_config = EasyDict(
        camera_type=camera_info.camera_type,
        camera_id=camera_info.camera_id,
        use_rgb=use_rgb,
        use_depth=use_depth,
        use_rec=use_rec,
        rgb_convention=rgb_convention,
    )

    # Print data publication information
    print("The node will publish the following data:")
    if use_rgb:
        print("- Color image")
    if use_depth:
        print("- Depth image")
    if use_rec:
        print("Note that Images are rectified with undistortion")

    # Create node configuration
    node_config = EasyDict(use_color=True, use_depth=True)
    if not use_rgb:
        node_config.use_color = False

    if not use_depth:
        node_config.use_depth = False

    if camera_info.camera_type == "rs":
        import pyrealsense2 as rs

        color_cfg = EasyDict(
            enabled=node_config.use_color,
            img_w=1280,
            img_h=720,
            img_format=rs.format.bgr8,
            fps=30,
        )

        depth_cfg = EasyDict(
            enabled=node_config.use_depth,
            img_w=1280,
            img_h=720,
            img_format=rs.format.z16,
            fps=30,
        )
        pc_cfg = EasyDict(enabled=False)
        camera_interface = RSInterface(
            device_id=camera_info.camera_id,
            color_cfg=color_cfg,
            depth_cfg=depth_cfg,
            pc_cfg=pc_cfg,
            serial=serial
        )
        print("got camera interface")

    # Create CameraRedisPubInterface
    camera2redis_pub_interface = CameraRedisPubInterface(
        camera_info=camera_info, redis_host=host, redis_port=port, #redis_password=redis_password
    )
    print("got camera2redis_pub_interface")

    return camera_interface, camera2redis_pub_interface, node_config, camera_config


def get_device_serial_num() -> List[str]:
    ctx: rs.context = rs.context()
    return [dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()]


def crop_and_resize_img(
    img: np.ndarray,
    do_crop: bool,
    crop_h: int,
    crop_w: int,
    do_resize: bool,
    resize_factor_x: float,
    resize_factor_y: float,
) -> np.ndarray:
    """Crops the center of the image to (crop_h, crop_w) and then resizes it by a given factor.

    Args:
        img (np.ndarray): Input image.
        crop_h (int): Height of the cropped region.
        crop_w (int): Width of the cropped region.
        resize_factor_x (float): Factor by which to resize the cropped image in the x direction.
        resize_factor_y (float): Factor by which to resize the cropped image in the y direction.

    Returns:
        np.ndarray: Cropped and resized image.
    """
    h, w = img.shape[:2]
    original_dim = len(img.shape)
    # print('original shape: ', img.shape)
    if do_crop:
        # Compute center crop coordinates
        start_x = max(0, w // 2 - crop_w // 2)
        end_x = min(w, start_x + crop_w)
        start_y = max(0, h // 2 - crop_h // 2)
        end_y = min(h, start_y + crop_h)

        img = img[start_y:end_y, start_x:end_x]

    if do_resize:
        # Resize with the given factor
        try:
            original_dtype = img.dtype
            # update to float32 for resize operation
            img = img.astype(np.float32)
            img = cv2.resize(
                img, (0, 0), fx=resize_factor_x, fy=resize_factor_y
            ).astype(original_dtype)
        except Exception as e:
            import ipdb

            ipdb.set_trace()
    # print('final shape: ', img.shape)

    new_dim = len(img.shape)
    if original_dim != new_dim:
        assert (
            new_dim == 2 and original_dim == 3
        ), f"original_dim: {original_dim}, new_dim: {new_dim} are not 2 as expected"
        img = np.expand_dims(img, axis=-1)
    return img


def save_collage(cur_obs, save_path="collage.png"):
    # Set up a transformation to convert tensors to images (to CPU and NumPy)
    to_numpy = transforms.ToPILImage()

    import matplotlib

    matplotlib.use("agg")
    # Create a grid to display the images in a collage format
    # We assume 3 rows and 4 columns to accommodate the images and masks
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))  # Adjust grid size if needed
    axes = axes.flatten()

    # Counter for image placement in the collage
    ax_idx = 0

    # Iterate through the dictionary and display the images, depth, and segmentation masks
    for key, tensor in cur_obs.items():
        # If the tensor represents image data (RGB, depth, or segmentation), we handle it differently
        if key in [
            "agentview_image",
            "eih_image",
            "agentview_depth",
            "robot0_eye_in_hand_depth",
            "agentview_segmentation_instance",
            "robot0_eye_in_hand_segmentation_instance",
        ]:
            # Squeeze the unnecessary batch and channel dimensions
            image = tensor.squeeze().cpu().numpy()

            # Handle different tensor shapes (ensure the image is 2D or 3D)
            if image.ndim == 3:  # RGB or depth (C, H, W)
                image = np.transpose(
                    image, (1, 2, 0)
                )  # Convert to (H, W, C) for image display
            elif image.ndim == 2:  # Grayscale or Segmentation masks (H, W)
                pass  # No change needed for 2D images (e.g., segmentation masks)

            # Plot the image in the collage
            axes[ax_idx].imshow(image)
            axes[ax_idx].set_title(key)
            axes[ax_idx].axis("off")  # Hide axis for clarity

            ax_idx += 1  # Move to the next subplot

    # Adjust layout and save the image collage
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)  # Save as a high-quality image
    plt.close()

def save_demo(demo: Dict[str, Any], save_path: str):
    # Open the HDF5 file in write mode
    with h5py.File(save_path, "w") as f:
        # Create a group for the demo
        demo_group = f.create_group("data/demo_0")
        # Save each dataset in the demo group
        demo_group.create_dataset("actions", data=np.array(demo["actions"]))
        # Create a group for observations
        observations_group = demo_group.create_group("observations")
        for key, value in demo["observations"].items():
            observations_group.create_dataset(key, data=np.array(value))
        demo_group.create_dataset("states", data=np.array(demo["states"]))
        demo_group.attrs["success"] = (
            demo["success"] if demo["success"] is not None else False
        )
        # Save failure type
        demo_group.attrs["failure_type"] = (
            demo["failure_type"] if demo["failure_type"] is not None else "None"
        )
    print(f"Saved demo to {save_path}")


import robosuite as suite
from robosuite.utils.input_utils import *
import cpgen_envs


def setup_env(env_name: str = "SquareRealBetterReal3pvCameraTableAlign", robot: str = "Panda_PandaUmiGripper"):
    """Initializes and resets the environment."""

    env = suite.make(
        env_name=env_name,
        robots=robot,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=20,
        base_types=None,
        camera_heights=720,
        camera_widths=1280,
        renderer="mujoco",
        camera_depths=True,
    )
    env.reset()
    action = np.zeros(7)
    # open gripper by default
    action[-2:] = -1
    for _ in range(10):
        env.step(action)
    return env

def main():
    env = setup_env()

    # yaml_config = OmegaConf.to_yaml(hydra_cfg, resolve=True)
    # cfg = EasyDict(yaml.safe_load(yaml_config))
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(cfg)
    # redis_host = "128.83.141.46"  # for internet
    redis_host = "172.16.0.1"
    save_dir = "rollouts"
    checkpoint_dir = "checkpoint-dir"
    robot_interface = FrankaInterface(
        os.path.join(config_root, "charmander.yml"), use_visualizer=False
    )
    ik_wrapper = IKWrapper()

    controller_cfg = YamlConfig(
        os.path.join(config_root, "joint-impedance-controller.yml")
    ).as_easydict()
    controller_type = "JOINT_IMPEDANCE"
    rotation_transformer = RotationTransformer("axis_angle", "rotation_6d")

    """ Data Collection Saving """
    from pathlib import Path

    folder = Path(save_dir)
    folder.mkdir(parents=True, exist_ok=True)

    experiment_id = 0
    for path in folder.glob("run*"):
        if not path.is_dir():
            continue
        try:
            folder_id = int(str(path).split("run")[-1])
            if folder_id > experiment_id:
                experiment_id = folder_id
        except BaseException:
            pass

    experiment_id += 1
    folder = str(folder / f"run{experiment_id}")
    os.makedirs(folder, exist_ok=True)

    """ End Data Collection Saving """

    """ Create data for saving """

    demo = {
        "actions": [],
        "observations": defaultdict(list),
    }

    eval_policy = get_policy(ckpt_path=checkpoint_dir)
    dir(eval_policy)

    # """ ======================= """

    renderer = ImageRenderer()
    renderer.start()

    camera_ids = [0, 1]
    serials = get_device_serial_num()

    cr_interfaces = {}
    for camera_id in camera_ids:
        cam_name = ""
        camera_ref = "rs_0"
        if camera_id == 0:
            cam_name = "agentview"
        else:
            camera_ref = "rs_1"
            cam_name = "robot0_eye_in_hand"
        camera_info = EasyDict({"camera_name": cam_name, "camera_id": camera_id})
        print("made easy dict")
        print(cam_name)

        cr_interface, camera2redis_pub_interface, node_config, camera_config = setup_camera_interface(camera_ref=camera_ref, serial=serials[camera_id])

        print("there")
        cr_interface.start()
        cr_interfaces[camera_id] = cr_interface

    time.sleep(3)

    time_index = 0

    os.makedirs("traj_episode/", exist_ok=True)

    last_obs = None
    model_config = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam_model = build_sam2(model_config, "../sam2/checkpoints/sam2.1_hiera_large.pt").to(
        "cuda"
    )
    mask_generator = SAM2AutomaticMaskGenerator(sam_model)
    MAX_HORIZON = 400
    for t in range(MAX_HORIZON):
        obs_images = []
        obs_dep = []

        agentview_image = None
        eye_in_hand_image = None

        agentview_dep = None
        eye_in_hand_dep = None
        agentview_seg = None
        eye_in_hand_seg = None

        for camera_id in camera_ids:
            imgs = cr_interfaces[camera_id].get_last_obs()
            img = imgs["color"]
            dep = imgs["depth"]

            # for size 84
            target_w = 90
            target_h = 160
            act_w = img.shape[0]
            act_h = img.shape[1]
            ratio_w = target_w / act_w
            ratio_h = target_h / act_h
            print("img type is ", img, type(img), img.dtype, np.max(img))
            resized_color_img = crop_and_resize_img(
                img, False, 0, 0, True, ratio_w, ratio_h
            )
            resized_depth_img = crop_and_resize_img(
                dep, False, 0, 0, True, ratio_w, ratio_h
            )

            resized_depth_img = resized_depth_img[..., None]
            resized_depth_img = resized_depth_img / 1000
            if camera_id == 0:
                agentview_image = resized_color_img
                obs_images.append(agentview_image)
                agentview_dep = resized_depth_img
                obs_dep.append(agentview_dep)
            elif camera_id == 1:
                eye_in_hand_image = resized_color_img
                obs_images.append(eye_in_hand_image)
                eye_in_hand_dep = resized_depth_img
                obs_dep.append(eye_in_hand_dep)
            else:
                raise NotImplementedError

        """ Creating Robomimic Observational Space """

        # agentview_seg = segment_black_table_sam2(agentview_image, mask_generator=mask_generator)
        eih_image, agentview_image = obs_images[0], obs_images[1]
        agentview_seg = segment_black_table(agentview_image, upper_black=[180, 180, 140])
        eih_seg = segment_black_table_sam(eih_image, black_v_threshold=120)

        seg_name = "agentvew_seg"
        img_name = "agentview"
        seg_path: Path = Path(save_dir) / f"{seg_name}.png"
        img_path: Path = Path(save_dir) / f"{img_name}.png"
        imageio.imwrite(str(seg_path), (agentview_seg * 255).squeeze(0))
        imageio.imwrite(str(img_path), agentview_image)

        agentview_image = np.transpose(obs_images[1], (2, 0, 1))
        agentview_image = np.float32(agentview_image) / 255.0

        # eih_image = np.transpose(obs_images[0], (2, 0, 1))
        # eih_image = np.float32(eih_image) / 255.0
        agentview_dep = np.transpose(
            obs_dep[1], (2, 0, 1)
        )  # does it come with a third channel? maybe we can add it in ourselves if necessary

        last_gripper_state = robot_interface._gripper_state_buffer[-1]

        last_state = robot_interface._state_buffer[-1]
        eef_pose_base = np.array(last_state.O_T_EE).reshape(4, 4).T
        eef_pos_base = eef_pose_base[:3, 3]
        eef_rotation_base = eef_rotation_world = eef_pose_base[:3, :3]
        # convert to world frame
        eef_pos_world = eef_pos_base + np.array(
            SquareRealBetterReal3pvCameraTableAlign.ROBOT_BASE_POS
        )
        from scipy.spatial.transform import Rotation as R
        eef_quat_world = R.from_matrix(eef_rotation_world).as_quat()

        gripper_states = np.array([last_gripper_state.width])
        # dividing gripper pos before feeding to model
        robot0_gripper_qpos = [gripper_states[0] / 2, -gripper_states[0] / 2]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        final_obs = cur_obs = {
            "agentview_segmentation_instance": torch.tensor(
                agentview_seg, device=device
            )
            .unsqueeze(0)
            .unsqueeze(0),
            "robot0_eye_in_hand_segmentation_instance": torch.tensor(
                eih_seg, device=device
            )
            .unsqueeze(0)
            .unsqueeze(0),
            "robot0_eef_pos": torch.tensor(eef_pos_world, device=device)
            .unsqueeze(0)
            .unsqueeze(0),
            "robot0_eef_quat": torch.tensor(eef_quat_world, device=device)[
                None, None, ...
            ],
            "robot0_gripper_qpos": torch.tensor(robot0_gripper_qpos, device=device)[
                None, None, ...
            ],
            # "robot0_qpos": torch.tensor(joint_states, device=device)[None, None, ...]
        }
        # final_obs = {
        #     # "agentview_depth": torch.zeros((1, 2, 1, 90, 160), device=device),
        #     # "robot0_eye_in_hand_depth": torch.zeros((1, 2, 1, 90, 160), device=device),
        #     "agentview_segmentation_instance": torch.zeros(
        #         (1, 2, 1, 90, 160), dtype=torch.uint8, device=device
        #     ),
        #     "robot0_eye_in_hand_segmentation_instance": torch.zeros(
        #         (1, 2, 1, 90, 160), dtype=torch.uint8, device=device
        #     ),
        #     "robot0_eef_pos": torch.zeros((1, 2, 3), device=device),
        #     "robot0_eef_quat": torch.zeros((1, 2, 4), device=device),
        #     "robot0_gripper_qpos": torch.zeros((1, 2, 2), device=device),
        #     # "robot0_qpos": torch.zeros((1, 2, 7), device=device)
        # }
        # upd_obs(last_obs, cur_obs, final_obs)

        for k, v in final_obs.items():
            print(k, v.shape, final_obs[k].shape)
            demo["observations"][k] = final_obs[k].cpu().detach().numpy()
        demo["observations"]["agentview_image"] = agentview_image
        demo["observations"]["eih_image"] = eih_image
        import ipdb;ipdb.set_trace()
        action = (
            eval_policy.predict_action(final_obs)["action_pred"].detach().cpu().numpy()
        )
        action = undo_transform_action(action, rotation_transformer)

        demo["actions"].append(action)
        from robosuite.utils.transform_utils import axisangle2quat, quat2mat

        action[0][..., :3] = action[0][..., :3] - np.array(
            SquareRealBetterReal3pvCameraTableAlign.ROBOT_BASE_POS
        )
        from scipy.spatial.transform import Rotation as R

        # R.from_rotvec(action[0][0, :3]).as_matrix()
        # quat2mat(axisangle2quat(action[0,0,:3]))

        """ =========== """
        for step in range(action.shape[1]):
            time_index += 1
            last_q = np.array(robot_interface.last_q)
            # action[0][step][:3] = action[0][step][:3] - np.array(
            #     SquareRealBetterReal3pvCameraTableAlign.ROBOT_BASE_POS
            # )
            target_world = action[0][step]
            target_world_position = target_world[:3]
            target_world_orientation = target_world[3:6]
            a = time.time()

            joint_traj, debug_info = ik_wrapper.ik_trajectory_to_target_pose(
                target_world_position,
                target_world_orientation,
                start_joint_positions=last_q.tolist(),
                num_points=1,
            )
            nn = time.time()
            if step == 0:
                save_collage(cur_obs, f"traj_episode/collage{time_index}.png")
            mm = time.time()
            b = time.time()
            joint_traj = ik_wrapper.interpolate_dense_traj(joint_traj)  # [::2]
            c = time.time()
            print(f"gripper action: {action[0][step]}")
            execute_ik_result(
                robot_interface,
                controller_type,
                controller_cfg,
                joint_traj,
                action[0][step][-1],
            )
            d = time.time()
            print(f"step 1 {b-a}. save_collage: {mm - nn}")
            print(f"step 2 {c-b}")
            print(f"step 3 {d-c}")
            
            # time.sleep(0.05)
        """ Saving Data at the End """
        last_state = robot_interface._state_buffer[-1]

        print(np.round(np.array(last_state.q), 3))
        # Get img info
        end_time = time.time_ns()
        print(action)
        print(t)

    for camera_id in camera_ids:
        np.savez(
            f"{folder}/testing_demo_camera_{camera_id}",
            data=np.array(data[f"camera_{camera_id}"]),
        )
        cr_interfaces[camera_id].stop()

    robot_interface.close()
    # Saving
    valid_input = False
    while not valid_input:
        try:
            save = input("Save or not? (enter 0 or 1)")
            save = bool(int(save))
            valid_input = True
        except:
            pass

    if not save:
        import shutil

        shutil.rmtree(f"{folder}")
        exit()

    if not cfg.intv:
        # Record success for eval mode
        valid_input = False
        while not valid_input:
            try:
                success = input("Success or fail? (enter 0 or 1)")
                success = bool(int(success))
                valid_input = True
            except:
                pass

        if success:
            np.savez(f"{folder}/success", data=np.array(success))

    # Printing dataset info
    import subprocess

    subprocess.run(
        [
            "python",
            "/home/huihanl/robot_control_ws/robot_infra/gprs/examples/count_intv.py",
            "--folder",
            cfg.save_dir,
            "--env",
            cfg.env,
        ]
    )


if __name__ == "__main__":
    main()
