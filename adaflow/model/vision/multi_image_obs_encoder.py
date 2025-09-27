from typing import Dict, List, Optional, Tuple, Union
import copy
import torch
import torch.nn as nn
import torchvision
from adaflow.model.vision.crop_randomizer import CropRandomizer
from adaflow.model.common.module_attr_mixin import ModuleAttrMixin
from adaflow.common.pytorch_util import dict_apply, replace_submodules

class Clamper(nn.Module):
    def __init__(self, max_value):
        super().__init__()
        self.max_value = max_value

    def forward(self, x):
        return torch.clamp(x, max=self.max_value)

class PixelWiseNormalizer(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x: torch.Tensor):
        mean = self.mean.unsqueeze(0).unsqueeze(0)
        std = self.std.unsqueeze(0).unsqueeze(0)

        return (x - mean) / std

class ApplySegmentationMask(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 2, H, W): channel 0 = depth, channel 1 = segmentation mask.
        # Assumes segmentation mask is binary (1 = segmented region).
        depth = x[..., 0:1, :, :]
        seg = x[..., 1:2, :, :]
        return depth * (1 - seg)

class MultiImageObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            rgb_model: Union[nn.Module, Dict[str,nn.Module]],
            resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            norm_args: Dict[str, Dict[str, int]] = None,
            random_crop: bool = True,
            fixed_crop: bool = False,  
            # Note: truthiness doesn't affect whether random_crop is used. Fixed crop
            # is used to always discard specific parts of the images, while random crop is used
            # as a data augmentation tool.
            fixed_crop_shape: Union[Tuple[int,int], Dict[str,tuple], None] = None,  # Add fixed_crop parameter
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            imagenet_norm: bool=False,
            depth_norm: bool=False,
            use_segmentation_input: bool=False,
            use_depth_input: bool=False,
            use_rgb_input: bool=True,
            keys_to_use: Optional[List] = None,
        ):
        """
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        """
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        depth_keys = list()
        segmentation_keys = list()
        segmented_depth_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        # handle sharing vision backbone
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map['rgb'] = rgb_model

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                if not use_rgb_input:
                    print(f"Skipping {key} as use_rgb_input is False")
                    continue

                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        # have provided model for each key
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(rgb_model)
                
                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features//16, 
                                num_channels=x.num_features)
                        )
                    key_model_map[key] = this_model
                
                # configure fixed crop logic
                input_shape = shape
                this_fixed_cropper = nn.Identity()
                if fixed_crop and fixed_crop_shape is not None:
                    if "DictConfig" in fixed_crop_shape.__class__.__name__:
                        h, w = fixed_crop_shape[key]
                    else:
                        h, w = fixed_crop_shape
                    this_fixed_cropper = torchvision.transforms.CenterCrop(size=(h, w))
                    input_shape = (1, h, w)

                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if "DictConfig" in resize_shape.__class__.__name__:
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(
                        size=(h,w)
                    )
                    input_shape = (shape[0],h,w)

                # configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False
                        )
                    else:
                        this_randomizer = torchvision.transforms.CenterCrop(
                            size=(h,w)
                        )
                # configure normalizer
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

                this_transform = nn.Sequential(this_fixed_cropper, this_resizer, this_normalizer, this_randomizer)
                key_transform_map[key] = this_transform
            elif type == 'depth':
                if not use_depth_input:
                    print(f"Skipping {key} as use_depth_input is False")
                    continue
                if keys_to_use is not None and key not in keys_to_use:
                    print(f"Skipping {key} as it is not in keys_to_use")
                    continue
                depth_keys.append(key)  # Add the key to the depth-specific list

                this_model = None

                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        # have provided model for each key
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(rgb_model)
                    this_model.conv1 = nn.Conv2d(
                        1, this_model.conv1.out_channels,
                        kernel_size=this_model.conv1.kernel_size,
                        stride=this_model.conv1.stride,
                        padding=this_model.conv1.padding,
                        bias=this_model.conv1.bias)
                
                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features//16, 
                                num_channels=x.num_features)
                        )
                    key_model_map[key] = this_model

                # configure fixed crop logic
                input_shape = shape
                this_fixed_cropper = nn.Identity()
                if fixed_crop and fixed_crop_shape is not None:
                    if "DictConfig" in fixed_crop_shape.__class__.__name__:
                        h, w = fixed_crop_shape[key]
                    else:
                        h, w = fixed_crop_shape
                    this_fixed_cropper = torchvision.transforms.CenterCrop(size=(h, w))
                    input_shape = (1, h, w)

                # Configure resize logic
                # input_shape = (1, *shape)
                input_shape = shape
                this_resizer = nn.Identity()  # Default to no resizing
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(size=(h, w))
                    input_shape = (1, h, w)
                
                # Configure randomization logic (e.g., random cropping or center cropping)
                this_randomizer = nn.Identity()  # Default to no randomization
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False  # No positional encoding for depth
                        )
                    else:
                        this_randomizer = torchvision.transforms.CenterCrop(size=(h, w))
                
                this_normalizer = nn.Identity()  # Default to no normalization
                this_clamper = nn.Identity()
                if depth_norm:
                    if norm_args is not None:
                        depth_norm_args = norm_args[key]
                        this_clamper = Clamper(max_value=depth_norm_args['clamp_max'])
                        this_normalizer = torchvision.transforms.Normalize(mean=[depth_norm_args['mean']], 
                                                                        std=[depth_norm_args['std']])
                # Combine all transformations
                this_transform = nn.Sequential(this_fixed_cropper, this_resizer, this_randomizer, this_clamper, this_normalizer)
                key_transform_map[key] = this_transform
            elif type == 'low_dim':
                low_dim_keys.append(key)
            elif type == "segmentation":
                if not use_segmentation_input:
                    print(f"Skipping {key} as use_segmentation_input is False")
                    continue
                if keys_to_use is not None and key not in keys_to_use:
                    print(f"Skipping {key} as it is not in keys_to_use")
                    continue
                segmentation_keys.append(key)
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        # have provided model for each key
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(rgb_model)
                    this_model.conv1 = nn.Conv2d(
                        1, this_model.conv1.out_channels,
                        kernel_size=this_model.conv1.kernel_size,
                        stride=this_model.conv1.stride,
                        padding=this_model.conv1.padding,
                        bias=this_model.conv1.bias)
                
                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features//16, 
                                num_channels=x.num_features)
                        )
                    key_model_map[key] = this_model

                # configure fixed crop logic
                input_shape = shape
                this_fixed_cropper = nn.Identity()
                if fixed_crop and fixed_crop_shape is not None:
                    if "DictConfig" in fixed_crop_shape.__class__.__name__:
                        h, w = fixed_crop_shape[key]
                    else:
                        h, w = fixed_crop_shape
                    this_fixed_cropper = torchvision.transforms.CenterCrop(size=(h, w))
                    input_shape = (1, h, w)

                # Configure resize logic
                # input_shape = (1, *shape)
                input_shape = shape
                this_resizer = nn.Identity()  # Default to no resizing
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(size=(h, w))
                    input_shape = (1, h, w)
                
                # Configure randomization logic (e.g., random cropping or center cropping)
                this_randomizer = nn.Identity()  # Default to no randomization
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False  # No positional encoding for depth
                        )
                    else:
                        this_randomizer = torchvision.transforms.CenterCrop(size=(h, w))
                
                this_normalizer = nn.Identity()  # Default to no normalization
                # Combine all transformations
                this_transform = nn.Sequential(this_fixed_cropper, this_resizer, this_normalizer, this_randomizer)
                key_transform_map[key] = this_transform
                # key_resizer_map[key] = this_resizer
                # key_randomizer_map[key] = this_randomizer
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # Verify all keys_to_use are valid
        if keys_to_use is not None:
            # First try to identify any segmented_depth keys from uncovered_keys
            uncovered_keys = set(keys_to_use) - set(rgb_keys) - set(depth_keys) - set(segmentation_keys)

            # Handle segmented_depth keys
            for key in list(uncovered_keys):
                if "_segmented_depth" in key:
                    segmented_depth_keys.append(key)
                    camera_name = key.replace("_segmented_depth", "")
                    depth_key = f"{camera_name}_depth"
                    seg_key = f"{camera_name}_segmentation_instance"
                    
                    # Check if required keys exist
                    if depth_key not in obs_shape_meta or seg_key not in obs_shape_meta:
                        raise ValueError(f"For {key}, need both {depth_key} and {seg_key} in obs_shape_meta")

                    this_model = None
                    # set shape to be same shape as depth and seg
                    shape = obs_shape_meta[depth_key]['shape']
                    # assert same as seg shape
                    assert shape == obs_shape_meta[seg_key]['shape'], f"Shape mismatch for {depth_key} and {seg_key}. Cannot use segmented depth."
                    if not share_rgb_model:
                        if isinstance(rgb_model, dict):
                            # have provided model for each key
                            this_model = rgb_model[key]
                        else:
                            assert isinstance(rgb_model, nn.Module)
                            # have a copy of the rgb model
                            this_model = copy.deepcopy(rgb_model)
                        this_model.conv1 = nn.Conv2d(
                            1, this_model.conv1.out_channels,
                            kernel_size=this_model.conv1.kernel_size,
                            stride=this_model.conv1.stride,
                            padding=this_model.conv1.padding,
                            bias=this_model.conv1.bias)
                    
                    if this_model is not None:
                        if use_group_norm:
                            this_model = replace_submodules(
                                root_module=this_model,
                                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                                func=lambda x: nn.GroupNorm(
                                    num_groups=x.num_features//16, 
                                    num_channels=x.num_features)
                            )
                        key_model_map[key] = this_model

                    # configure fixed crop logic
                    input_shape = shape
                    this_fixed_cropper = nn.Identity()
                    if fixed_crop and fixed_crop_shape is not None:
                        if "DictConfig" in fixed_crop_shape.__class__.__name__:
                            h, w = fixed_crop_shape[key]
                        else:
                            h, w = fixed_crop_shape
                        this_fixed_cropper = torchvision.transforms.CenterCrop(size=(h, w))
                        input_shape = (1, h, w)

                    # Configure resize logic
                    # input_shape = (1, *shape)
                    input_shape = shape
                    this_resizer = nn.Identity()  # Default to no resizing
                    if resize_shape is not None:
                        if isinstance(resize_shape, dict):
                            h, w = resize_shape[key]
                        else:
                            h, w = resize_shape
                        this_resizer = torchvision.transforms.Resize(size=(h, w))
                        input_shape = (1, h, w)
                    
                    # Configure randomization logic (e.g., random cropping or center cropping)
                    this_randomizer = nn.Identity()  # Default to no randomization
                    if crop_shape is not None:
                        if isinstance(crop_shape, dict):
                            h, w = crop_shape[key]
                        else:
                            h, w = crop_shape
                        if random_crop:
                            this_randomizer = CropRandomizer(
                                input_shape=input_shape,
                                crop_height=h,
                                crop_width=w,
                                num_crops=1,
                                pos_enc=False  # No positional encoding for depth
                            )
                        else:
                            this_randomizer = torchvision.transforms.CenterCrop(size=(h, w))
                    
                    this_normalizer = nn.Identity()  # Default to no normalization
                    this_clamper = nn.Identity()
                    if depth_norm:
                        if norm_args is not None:
                            depth_norm_args = norm_args[depth_key]
                            this_clamper = Clamper(max_value=depth_norm_args['clamp_max'])
                            this_normalizer = torchvision.transforms.Normalize(mean=[depth_norm_args['mean']], 
                                                                            std=[depth_norm_args['std']])
                    # Combine all transformations; should be okay to get segmented depth first and then normalize and then clamp.
                    this_transform = nn.Sequential(ApplySegmentationMask(), this_fixed_cropper, this_resizer, this_randomizer, this_clamper, this_normalizer)
                    key_transform_map[key] = this_transform
                    # Remove from uncovered keys since we've handled it
                    uncovered_keys.remove(key)

            # Raise error for any remaining uncovered keys
            if len(uncovered_keys) > 0:
                raise ValueError(f"The following keys in keys_to_use are not handled: {uncovered_keys}")

        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        depth_keys = sorted(depth_keys)
        segmentation_keys = sorted(segmentation_keys)
        segmented_depth_keys = sorted(segmented_depth_keys)
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.depth_keys = depth_keys
        self.segmentation_keys = segmentation_keys
        self.segmented_depth_keys = segmented_depth_keys
        self.key_shape_map = key_shape_map


    def forward(self, obs_dict):
        batch_size = None
        features = list()
        # process rgb input
        if self.share_rgb_model:
            # pass all rgb obs to rgb model
            imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                imgs.append(img)
            # (N*B,C,H,W)
            imgs = torch.cat(imgs, dim=0)
            # (N*B,D)
            feature = self.key_model_map['rgb'](imgs)
            # (N,B,D)
            feature = feature.reshape(-1,batch_size,*feature.shape[1:])
            # (B,N,D)
            feature = torch.moveaxis(feature,0,1)
            # (B,N*D)
            feature = feature.reshape(batch_size,-1)
            features.append(feature)
        else:
            # run each rgb obs to independent models
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                try:
                    assert img.shape[1:] == self.key_shape_map[key]
                except AssertionError as e:
                    import ipdb; ipdb.set_trace()
                    print(f"key: {key}")
                    print(f"img.shape: {img.shape}")
                    print(f"key_shape_map[key]: {self.key_shape_map[key]}")
                    raise e
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                features.append(feature)
        for key in self.depth_keys:
            depth = obs_dict[key]
            if batch_size is None:
                batch_size = depth.shape[0]
            else:
                assert batch_size == depth.shape[0]
            if depth.shape[1:] != self.key_shape_map[key]:
                depth = depth.unsqueeze(1)
            assert depth.shape[1:] == self.key_shape_map[key]
            depth = self.key_transform_map[key](depth)  # Apply transformations
            feature = self.key_model_map[key](depth)
            features.append(feature)

        for key in self.segmentation_keys:
            segmentation = obs_dict[key]
            if batch_size is None:
                batch_size = segmentation.shape[0]
            else:
                assert batch_size == segmentation.shape[0]
            if segmentation.shape[1:] != self.key_shape_map[key]:
                segmentation = segmentation.unsqueeze(1)
            assert segmentation.shape[1:] == self.key_shape_map[key]
            segmentation = self.key_transform_map[key](segmentation)
            feature = self.key_model_map[key](segmentation)
            features.append(feature)

        for key in self.segmented_depth_keys:
            # should be "agentview_segmented_depth"
            # take the segmentation + depth
            depth_key = f"{key.replace('_segmented_depth', '')}_depth"
            segmentation_key = f"{key.replace('_segmented_depth', '')}_segmentation_instance"
            depth = obs_dict[depth_key]
            segmentation = obs_dict[segmentation_key]
            # combine depth and segmentation into single image
            combined = torch.stack([depth.squeeze(1), segmentation.squeeze(1)], dim=1)
            segmented_depth = self.key_transform_map[key](combined)
            feature = self.key_model_map[key](segmented_depth)
            features.append(feature)

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)
        
        # concatenate all features
        result = torch.cat(features, dim=-1)
        return result
    
    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape
