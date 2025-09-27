## Training on Ground Truth Depth Values

  

**1.** Download Dataset

You can follow Step #2 [here](https://github.com/kevin-thankyou-lin/adaflow/tree/main#modifications-for-robosuite-v15)

**2.** Convert Raw Data to Depth data

You can run this [script](dataset_states_to_obs.py) with the following command:

```bash

python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name depth.hdf5 \

--done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 --depth

```
It might help to visualize the data/get some insights once it's downloaded from [here](https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/scripts/playback_dataset.py) or [here](https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/scripts/get_dataset_info.py). 

**3.** Adjust parameters

Go to the task file you're planning to use (e.g. `adaflow/config/task/square_image_abs.yaml`), and update the `dataset_path`, to the `output_name` file you used with `dataset_states_to_obs.py`.

Also update `shapes_meta` to use the corresponding `depth` keys and types instead of `image` or `rgb`. Modify `render_obs_key` in the `env_runner` section to use `agentview_depth`, and `camera_depths` to True. Lastly, make sure to modify the default task of [training config](../adaflow/config/train_diffusion_unet_ddpm_image_workspace_robomimic.yaml) to `square_depth_abs`.

**4.** Train the Policy

Follow the command on Step #3 [here](https://github.com/kevin-thankyou-lin/adaflow/tree/main#modifications-for-robosuite-v15), using the appropriate task, `square_depth_abs`


## Training on Ground Truth Depth + Segmentation Mask Values

**1.** Download dataset

Follow step 1 from above.

**2.** Convert Raw Data to Depth + Segmentation data

You can run this [script](dataset_states_to_obs.py) with the following command:

```bash

python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name depth-seg.hdf5 \
--done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 --depth --segmentation instance --compress --exclude-next-obs

```
It might help to visualize the data/get some insights once it's downloaded from [here](https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/scripts/playback_dataset.py) or [here](https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/scripts/get_dataset_info.py). 

**3.** Adjust parameters

See `adaflow/config/task/square_depth_seg_abs.yaml` and `adaflow/config/train_diffusion_unet_ddpm_image_workspace_robomimic_depth_seg.yaml`.

Note: we have hardcoded !! values for depth clamping and normalization in the `...robomimic_depth_seg.yaml` file. Please adapt as needed. Note: normalization values
should we computed *after* the clamping is done.
