"""

Reference: https://github.com/mihdalal/manipgen/blob/e39e6147d17a3870f11ff41532b2e641a65acbab/manipgen/utils/obs_utils.py#L51

Uses depth augmentation to get noise depth images for training 

"""
from typing import Dict, List
import torch
import torch.nn.functional as F
import numpy as np
import h5py
from tqdm import tqdm
import zarr
import os
import shutil
import copy
import json
import hashlib
from filelock import FileLock
from threadpoolctl import threadpool_limits
import concurrent.futures
import multiprocessing
from omegaconf import OmegaConf
from adaflow.common.pytorch_util import dict_apply
from adaflow.dataset.base_dataset import BaseImageDataset, LinearNormalizer
from adaflow.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from adaflow.model.common.rotation_transformer import RotationTransformer
from adaflow.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from adaflow.common.replay_buffer import ReplayBuffer
from adaflow.common.sampler import SequenceSampler, get_val_mask
from torchvision.transforms.v2 import GaussianBlur
import random
from adaflow.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats,
    get_depth_normalizer_to_unit_range
)
import imageio
register_codecs()

class RobomimicReplayImageDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            abs_action=False,
            rotation_rep='rotation_6d', # ignored when abs_action=False
            use_legacy_normalizer=False,
            use_cache=False,
            seed=42,
            val_ratio=0.0,
            norm_args: Dict[str, Dict[str, int]] = None,
            use_depth_aug=False,
            use_seg_aug=False,
            depth_handler_cfg: dict = None,
            debug=False
        ):

        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep=rotation_rep)

        replay_buffer = None
        if use_cache:
            cache_zarr_path = dataset_path + '.zarr.zip'
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        # store = zarr.DirectoryStore(cache_zarr_path)
                        replay_buffer = _convert_robomimic_to_replay(
                            store=zarr.MemoryStore(), 
                            shape_meta=shape_meta, 
                            dataset_path=dataset_path, 
                            abs_action=abs_action, 
                            rotation_transformer=rotation_transformer)
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _convert_robomimic_to_replay(
                store=zarr.MemoryStore(), 
                shape_meta=shape_meta, 
                dataset_path=dataset_path, 
                abs_action=abs_action, 
                rotation_transformer=rotation_transformer)

        rgb_keys = list()
        lowdim_keys = list()
        depth_keys = list()
        seg_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
            elif type == 'depth':
                depth_keys.append(key)
            elif type == 'segmentation':
                seg_keys.append(key)
        
        # for key in rgb_keys:
        #     replay_buffer[key].compressor.numthreads=1

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys + depth_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        # per_pixel_mean, per_pixel_std = get_stats_per_pixel(norm_args, replay_buffer)
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.depth_keys = depth_keys
        self.seg_keys = seg_keys
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer
        # self.per_pixel_mean = per_pixel_mean
        # self.per_pixel_std = per_pixel_std
        self.use_depth_aug = use_depth_aug
        self.depth_handler_cfg = depth_handler_cfg
        self.use_seg_aug = use_seg_aug
        self.debug = debug

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer['action'])
        if self.abs_action:
            if stat['mean'].shape[-1] > 10:
                # dual arm
                this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
            else:
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
            
            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            # already normalized
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer['action'] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        for key in self.depth_keys:
            normalizer[key] = get_depth_normalizer_to_unit_range()

        for key in self.seg_keys:
            stat = array_to_stats(self.replay_buffer[key])
            for stat_key, stat_val in stat.items():
                stat[stat_key] = np.array(stat_val, dtype=np.float32)
            normalizer[key] = get_identity_normalizer_from_stat(stat)

        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key][T_slice],-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]
        for key in self.depth_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]
        for key in self.seg_keys:
            obs_dict[key] = data[key][T_slice].astype(np.uint8)
            del data[key]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        if self.use_depth_aug and self.depth_handler_cfg is not None:
            add_depth_aug(torch_data['obs'], self.depth_keys, self.depth_handler_cfg, self.debug)
            
        if self.use_seg_aug:
            for key in self.seg_keys:
                augment_segmentation_mask(
                    torch_data['obs'][key],
                    p_flip=0.003,
                    p_stick=0.001,
                    stick_length_range=(5, 12),
                    stick_width_range=(1, 2), debug=False
                )
        return torch_data


def _convert_actions(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1,2,7)
            is_dual_arm = True

        pos = raw_actions[...,:3]
        rot = raw_actions[...,3:6]
        gripper = raw_actions[...,6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([
            pos, rot, gripper
        ], axis=-1).astype(np.float32)
    
        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1,20)
        actions = raw_actions
    return actions

def add_depth_aug(torch_data: Dict, depth_keys: List, depth_handler_cfg: Dict, debug: bool = False):
    preprocessed_depth = None
    if debug:
        preprocessed_depth = torch_data[depth_keys[0]].shape
    for key in depth_keys:
        depth_obs = torch_data[key]
        # Assume add_depth_noise() is your augmentation function that takes the depth tensor
        # and the configuration. It will use the parameters from depth_handler_cfg.
        torch_data[key] = add_depth_noise(
            depth_obs, 
            depth_handler_cfg['depth_handler']['augmentation'],
            device=depth_obs.device
        )
    if debug:
        save_depth_map(preprocessed_depth, "test1.jpg")
        save_depth_map(torch_data[depth_keys[0]][1], "test2.jpg")


def _convert_robomimic_to_replay(store, shape_meta, dataset_path, abs_action, rotation_transformer, 
        n_workers=None, max_inflight_tasks=None):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    lowdim_keys = list()
    depth_keys = list()
    seg_keys = list()

    # construct compressors and chunks
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        shape = attr['shape']
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            rgb_keys.append(key)
        elif type == 'low_dim':
            lowdim_keys.append(key)
        elif type == 'depth':
            depth_keys.append(key)
        elif type == 'segmentation':
            seg_keys.append(key)
    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)
    
    with h5py.File(dataset_path, "r") as file:
        # count total steps
        demos = file['data']
        episode_ends = list()
        prev_end = 0
        for i in range(len(demos)):
            try:
                demo = demos[f'demo_{i}']
            except KeyError as e:
                print(f"Skipping demo_{i} due to error: {e}")
                continue
            episode_length = demo['actions'].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
        n_steps = episode_ends[-1]
        episode_starts = [0] + episode_ends[:-1]
        _ = meta_group.array('episode_ends', episode_ends, 
            dtype=np.int64, compressor=None, overwrite=True)

        # save lowdim data
        for key in tqdm(lowdim_keys + ['action'], desc="Loading lowdim data"):
            data_key = 'obs/' + key
            if key == 'action':
                data_key = 'actions'
            this_data = list()
            for i in range(len(demos)):
                demo = demos[f'demo_{i}']
                try:
                    this_data.append(demo[data_key][:].astype(np.float32))
                except Exception as e:
                    print(f"Skipping demo_{i} due to error: {e}")
                    import ipdb; ipdb.set_trace()
            this_data = np.concatenate(this_data, axis=0)
            if key == 'action':
                this_data = _convert_actions(
                    raw_actions=this_data,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer
                )
                assert this_data.shape == (n_steps,) + tuple(shape_meta['action']['shape'])
            else:
                assert this_data.shape == (n_steps,) + tuple(shape_meta['obs'][key]['shape'])
            _ = data_group.array(
                name=key,
                data=this_data,
                shape=this_data.shape,
                chunks=this_data.shape,
                compressor=None,
                dtype=this_data.dtype
            )
        
        with tqdm(total=n_steps * len(depth_keys), desc="Loading depth data", mininterval=1.0) as pbar:
            for key in depth_keys:
                data_key = 'obs/' + key
                shape = tuple(shape_meta['obs'][key]['shape'])
                c, h, w = shape
                print(f"Loading depth data for key: {key}")
                print(f"shape: {shape}")
                # this_compressor = None # Might be smart to consider a compressor later if memory becomes a problem...
                depth_arr = data_group.require_dataset(
                    name=key,
                    shape=(n_steps, h, w),
                    chunks=(1, h, w),
                    compressor=None,
                    dtype=np.float32
                )
                for episode_idx in range(len(demos)):
                    demo = demos[f'demo_{episode_idx}']
                    hdf5_arr = demo['obs'][key]
                    for hdf5_idx in range(hdf5_arr.shape[0]):
                        zarr_idx = episode_starts[episode_idx] + hdf5_idx
                        depth_arr[zarr_idx] = hdf5_arr[hdf5_idx][..., 0]
                        pbar.update(1)

        with tqdm(total=n_steps * len(seg_keys), desc="Loading segmentation data", mininterval=1.0) as pbar:
            for key in seg_keys:
                data_key = 'obs/' + key
                shape = tuple(shape_meta['obs'][key]['shape'])
                c, h, w = shape
                print(f"Loading segmentation data for key: {key}")
                print(f"shape: {shape}")
                # this_compressor = None # Might be smart to consider a compressor later if memory becomes a problem...
                seg_arr = data_group.require_dataset(
                    name=key,
                    shape=(n_steps, h, w),
                    chunks=(1, h, w),
                    compressor=None,
                    dtype=np.uint8
                )
                for episode_idx in range(len(demos)):
                    demo = demos[f'demo_{episode_idx}']
                    hdf5_arr = demo['obs'][key]
                    for hdf5_idx in range(hdf5_arr.shape[0]):
                        zarr_idx = episode_starts[episode_idx] + hdf5_idx
                        seg_arr[zarr_idx] = hdf5_arr[hdf5_idx][..., 0]
                        pbar.update(1)
        
        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                # make sure we can successfully decode
                _ = zarr_arr[zarr_idx]
                return True
            except Exception as e:
                print(f"Failed to encode image: {e}")
                return False

        with tqdm(total=n_steps*len(rgb_keys), desc="Loading image data", mininterval=1.0) as pbar:
            # one chunk per thread, therefore no synchronization needed
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = set()
                for key in rgb_keys:
                    data_key = 'obs/' + key
                    shape = tuple(shape_meta['obs'][key]['shape'])
                    c,h,w = shape
                    print(f"Loading image data for key: {key}")
                    print(f"shape: {shape}")
                    this_compressor = Jpeg2k(level=50)
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps,h,w,c),
                        chunks=(1,h,w,c),
                        compressor=this_compressor,
                        dtype=np.uint8
                    )
                    for episode_idx in range(len(demos)):
                        demo = demos[f'demo_{episode_idx}']
                        hdf5_arr = demo['obs'][key]
                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                # limit number of inflight tasks
                                completed, futures = concurrent.futures.wait(futures, 
                                    return_when=concurrent.futures.FIRST_COMPLETED)
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError('Failed to encode image!')
                                pbar.update(len(completed))

                            zarr_idx = episode_starts[episode_idx] + hdf5_idx
                            futures.add(
                                executor.submit(img_copy, 
                                    img_arr, zarr_idx, hdf5_arr, hdf5_idx))
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError('Failed to encode image!')
                pbar.update(len(completed))

    replay_buffer = ReplayBuffer(root)
    return replay_buffer

def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

    

def get_stats_per_pixel(norm_args, replay_buffer):
    per_pixel_mean = {}
    per_pixel_std = {}
    for key in replay_buffer.keys():
        if "depth" in key:
            clamp_max = norm_args[key]['clamp_max']
            zarr_array = replay_buffer[key]
            data = zarr_array[:]
            clamped_data = np.clip(data, None, clamp_max)

            mean = np.mean(clamped_data, axis=0)
            std = np.std(clamped_data, axis=0)

            per_pixel_mean[key] = mean
            per_pixel_std[key] = std
    return per_pixel_mean, per_pixel_std


def augment_segmentation_mask(
    mask: torch.Tensor,
    p_flip: float = 0.002,
    p_stick: float = 0.0025,
    stick_length_range: tuple = (5, 18),
    stick_width_range: tuple = (1, 3),
    debug: bool = False
) -> None:
    """
    Apply augmentations to a segmentation mask in-place to simulate wire occlusions and artifacts.
    
    Args:
        mask (torch.Tensor): Input segmentation mask (1, H, W) with integer labels. Modified in-place. uint8.
        p_flip (float): Probability of flipping a pixel.
        p_stick (float): Probability of adding stick artifacts.
        stick_length_range (tuple): Range of stick artifact lengths.
        stick_width_range (tuple): Range of stick artifact widths.
    
    Returns:
        None (modifies the input mask in-place).
    """
    assert mask.ndim == 3 and mask.shape[0] == 1, "Mask should have shape (1, H, W)"
    
    if debug:
        # Save input mask for visualization
        imageio.imwrite('input_mask.png', mask[0].cpu().numpy() * 255)
    _, h, w = mask.shape

    # Apply dropout
    flip_mask = (torch.rand((h, w), device=mask.device) < p_flip)
    mask[:, flip_mask] = 1 - mask[:, flip_mask]

    if debug:
        imageio.imwrite('output_mask.png', mask[0].cpu().numpy() * 255)

    # Randomly determine the number of sticks
    num_sticks = torch.randint(0, int(h * w * p_stick) + 1, (1,), device=mask.device).item()

    y_coords, x_coords = torch.meshgrid(torch.arange(h, device=mask.device), 
                                        torch.arange(w, device=mask.device), indexing='ij')

    for _ in range(num_sticks):
        # Select a random start point
        x_start = torch.randint(0, w, (1,), device=mask.device).item()
        y_start = torch.randint(0, h, (1,), device=mask.device).item()
        
        # Select a random direction
        angle = torch.rand(1, device=mask.device).item() * 2 * np.pi
        
        # Select a random length within the given range
        length = torch.randint(*stick_length_range, (1,), device=mask.device).item()
        
        # Compute end point within max_length
        x_end = x_start + int(length * np.cos(angle))
        y_end = y_start + int(length * np.sin(angle))

        # Clamp to valid image bounds
        x_end = max(0, min(w - 1, x_end))
        y_end = max(0, min(h - 1, y_end))

        # Compute unit direction vector
        dx = torch.tensor(x_end - x_start, dtype=torch.float32, device=mask.device)
        dy = torch.tensor(y_end - y_start, dtype=torch.float32, device=mask.device)
        norm = torch.sqrt(dx**2 + dy**2 + 1e-6)  # Avoid division by zero
        direction_x = dx / norm
        direction_y = dy / norm

        # Compute projection of each pixel onto the stick vector
        proj_length = (x_coords - x_start) * direction_x + (y_coords - y_start) * direction_y

        # Ensure projections are within [0, length]
        within_length = (proj_length >= 0) & (proj_length <= length)

        # Select random width
        width = torch.randint(*stick_width_range, (1,), device=mask.device).item()

        # Compute perpendicular distance to the stick
        perp_dist = torch.abs((x_coords - x_start) * direction_y - (y_coords - y_start) * direction_x)

        # Ensure the pixels are within width bounds
        within_width = perp_dist <= width / 2

        # Final stick mask
        stick_mask = within_length & within_width

        # Randomly set sticks to either 0 (background) or 1 (artifact)
        stick_value = torch.randint(0, 2, (1,), device=mask.device).item()
        mask[:, stick_mask] = stick_value

    if debug:
        imageio.imwrite('stick_mask.png', mask[0].cpu().numpy() * 255)



def depth_warping(depths, std=0.5, prob=0.8, device=None):
    """
    Applies edge noise via depth warping.
    For each pixel, with probability `prob` a random shift from N(0, std) is added.

    Args:
        depths (torch.Tensor): Depth tensor of shape (T, H, W) (single channel assumed).
        std (float): Standard deviation for the random shift.
        prob (float): Probability of applying a shift for each pixel.
        device (torch.device): Computation device.
        
    Returns:
        torch.Tensor: Depth tensor after applying bilinear interpolation with the shifted grid, shape (T, H, W).
    """
    device = device or depths.device
    # If input is (T, H, W), add a channel dimension for grid_sample.
    if depths.dim() == 3:
        depths = depths.unsqueeze(1)  # Now (T, 1, H, W)
        added_channel = True
    else:
        added_channel = False

    T, C, H, W = depths.shape  # Here, C should be 1.
    
    # Generate per-pixel Gaussian shifts (for x and y)
    gaussian_shifts = torch.normal(mean=0, std=std, size=(T, H, W, 2), device=device)
    # Create a mask to decide per-pixel if a shift is applied.
    apply_mask = (torch.rand(T, H, W, 1, device=device) < prob).float()
    gaussian_shifts = gaussian_shifts * apply_mask

    # Create a grid of original pixel coordinates (shape: T, H, W, 2)
    xx = torch.linspace(0, W - 1, W, device=device).view(1, 1, W).expand(T, H, W)
    yy = torch.linspace(0, H - 1, H, device=device).view(1, H, 1).expand(T, H, W)
    grid = torch.stack((xx, yy), dim=3)  # (T, H, W, 2)

    # Add the Gaussian shifts to the grid.
    grid = grid + gaussian_shifts

    # Normalize grid coordinates to [-1, 1] for grid_sample.
    grid[..., 0] = (grid[..., 0] / (W - 1)) * 2 - 1
    grid[..., 1] = (grid[..., 1] / (H - 1)) * 2 - 1

    # Warp the depth maps using bilinear interpolation.
    depths_interp = F.grid_sample(depths, grid, mode='bilinear', padding_mode='border', align_corners=True)
    
    # Remove the added channel if necessary.
    if added_channel:
        depths_interp = depths_interp.squeeze(1)
    return depths_interp

def generate_mask(T, H, W, device, holes_cfg):
    """
    Generates a random mask to simulate holes in the depth map.
    The mask is computed by generating random noise, applying Gaussian blur,
    normalizing, and then thresholding.

    Args:
        T (int): Number of depth maps.
        H (int): Height of each depth map.
        W (int): Width of each depth map.
        device (torch.device): Computation device.
        holes_cfg (dict): Configuration for holes augmentation.
        
    Returns:
        torch.Tensor: Boolean mask tensor of shape (T, 1, H, W).
    """
    # Generate random noise in [0,1] with a channel dimension.
    noise = torch.rand(T, 1, H, W, device=device)

    # Choose an odd kernel size from the provided range.
    k_lower = holes_cfg['kernel_size_lower']
    k_upper = holes_cfg['kernel_size_upper']
    possible_ks = list(range(k_lower, k_upper + 1, 2))
    kernel_size = random.choice(possible_ks)

    # Apply Gaussian blur using the repo's implementation.
    sigma_lower = holes_cfg['sigma_lower']
    sigma_upper = holes_cfg['sigma_upper']
    blur = GaussianBlur(kernel_size=kernel_size, sigma=(sigma_lower, sigma_upper))
    noise = blur(noise)

    # Normalize the blurred noise to [0, 1].
    noise = (noise - noise.min()) / (noise.max() - noise.min())

    # Sample a threshold uniformly from U(thresh_lower, thresh_upper)
    thresh_lower = holes_cfg['thresh_lower']
    thresh_upper = holes_cfg['thresh_upper']
    thresh = torch.rand(T, 1, 1, 1, device=device) * (thresh_upper - thresh_lower) + thresh_lower

    mask = noise > thresh
    return mask

def add_depth_noise(depth_tensor, aug_cfg, device=None):
    """
    Applies depth augmentations as described in the paper:
      1. Edge noise (via depth warping)
      2. Random holes (by zeroing out pixels based on a blurred mask)
      
    Args:
        depth_tensor (torch.Tensor): Depth tensor of shape (T, H, W) (single channel assumed).
        aug_cfg (dict): Augmentation configuration (e.g. cfg.depth_handler.augmentation).
        device (torch.device): Computation device.
        
    Returns:
        torch.Tensor: Augmented depth tensor of shape (T, H, W).
    """
    device = device or depth_tensor.device
    T, H, W = depth_tensor.shape
    obs = depth_tensor.clone().to(device)

    # 1. Apply edge noise (depth warping) if enabled.
    if aug_cfg.get('depth_warping', {}).get('enabled', False):
        std = aug_cfg['depth_warping'].get('std', 0.5)
        prob = aug_cfg['depth_warping'].get('prob', 0.8)
        obs = depth_warping(obs, std=std, prob=prob, device=device)

    # 2. Apply random holes if enabled.
    if aug_cfg.get('holes', {}).get('enabled', False):
        holes_prob = aug_cfg['holes'].get('prob', 0.5)
        # Generate the mask for T samples.
        mask = generate_mask(T, H, W, device, aug_cfg['holes'])
        # For each depth map, decide whether to apply holes using holes_prob.
        sample_apply = (torch.rand(T, 1, 1, 1, device=device) < holes_prob)
        # Apply holes only where sample_apply is True.
        mask = mask & sample_apply.bool()
        # Squeeze mask from (T, 1, H, W) to (T, H, W) to match obs.
        mask = mask.squeeze(1)
        # Zero out pixels where the mask is True.
        obs[mask] = 0.0

    return pixel_dropout(obs, 1/50)

            

def save_depth_map(depth_tensor, filename):
    """
    Saves a [H, W] depth map tensor as an 8-bit image (e.g., JPEG) using imageio.
    Normalizes the depth map to the range [0, 255].
    
    Args:
        depth_tensor (torch.Tensor): Depth map of shape [H, W].
        filename (str): File path to save the image.
    """
    # Convert tensor to numpy array
    depth_np = depth_tensor.detach().cpu().numpy()
    
    # Normalize to [0, 255]
    depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min()) * 255
    depth_uint8 = depth_norm.astype(np.uint8)
    
    # Save as JPEG (or PNG)
    imageio.imwrite(filename, depth_uint8)

def pixel_dropout(depth_tensor, drop_prob=1/30):
    """
    Sets each pixel in the depth_tensor to 0 with probability drop_prob.
    
    Args:
        depth_tensor (torch.Tensor): A depth map tensor. It can be 2D ([H, W]) or 3D ([T, H, W]).
        drop_prob (float): The probability of dropping each pixel (default 1/30).
        
    Returns:
        torch.Tensor: The depth map with random pixels set to 0.
    """
    # Create a mask with the same shape as the depth tensor:
    # True with probability drop_prob, False otherwise.
    mask = torch.rand_like(depth_tensor) < drop_prob
    
    # Clone the depth tensor to avoid in-place modification.
    out = depth_tensor.clone()
    out[mask] = 0.0
    return out
