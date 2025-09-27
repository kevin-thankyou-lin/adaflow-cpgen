from typing import Dict, Callable, Tuple
import numpy as np
# from diffusion_policy.common.cv2_util import get_image_transform
import cv2

def get_real_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        ) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            this_imgs_in = env_obs[key]
            t,hi,wi,ci = this_imgs_in.shape
            co,ho,wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi,hi), 
                    output_res=(wo,ho), 
                    bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
        elif type == 'low_dim':
            this_data_in = env_obs[key]
            if 'pose' in key and shape == (2,):
                # take X,Y coordinates
                this_data_in = this_data_in[...,[0,1]]
            obs_dict_np[key] = this_data_in
    return obs_dict_np


def get_real_obs_resolution(
        shape_meta: dict
        ) -> Tuple[int, int]:
    out_res = None
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            co,ho,wo = shape
            if out_res is None:
                out_res = (wo, ho)
            assert out_res == (wo, ho)
    return out_res


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
            img = cv2.resize(img, (0, 0), fx=resize_factor_x, fy=resize_factor_y).astype(original_dtype)
        except Exception as e:
            import ipdb; ipdb.set_trace()
    # print('final shape: ', img.shape)

    new_dim = len(img.shape)
    if original_dim != new_dim:
        assert new_dim == 2 and original_dim == 3, f"original_dim: {original_dim}, new_dim: {new_dim} are not 2 as expected"
        img = np.expand_dims(img, axis=-1)
    return img
