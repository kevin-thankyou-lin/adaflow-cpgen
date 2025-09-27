from typing import List, Tuple, Dict, Iterator, Optional
from pathlib import Path
from dataclasses import dataclass
from flask import Flask, Response
import numpy as np
import cv2
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
import argparse
import robosuite as suite  # Ensure robosuite is installed
import cpgen_envs  # Assumed dependency

@dataclass
class StreamConfig:
    blend_frames: bool
    blend_ratio: float
    env_placement_bounds: Dict[str, Tuple[float, float, float, float]]
    env_image: Optional[np.ndarray] = None

def get_device_serial_numbers() -> List[str]:
    ctx = rs.context()
    return [dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()]

def setup_pipeline(serial: str) -> rs.pipeline:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline

# Set up the RealSense pipeline.
serials = get_device_serial_numbers()
assert serials, "No RealSense devices found."
# pipeline = setup_pipeline(serials[1])
pipeline = setup_pipeline(serials[1])
profile = pipeline.get_active_profile()
color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
intr = color_profile.get_intrinsics()  # type: rs.intrinsics

# Extrinsics from the robot (world) frame.
POS_IN_ROBOT_FRAME = np.array([1.15, -0.042, 0.55])
EULER_XYZ = np.array([-135, 0, 90])
rot_mat = R.from_euler("xyz", EULER_XYZ, degrees=True).as_matrix()

def project_polygon_from_world(x_min: float, x_max: float,
                               y_min: float, y_max: float,
                               z_fixed: float,
                               intr: rs.intrinsics,
                               rot_mat: np.ndarray,
                               t: np.ndarray) -> np.ndarray:
    corners_world = np.array([
        [x_min, y_min, z_fixed],
        [x_min, y_max, z_fixed],
        [x_max, y_max, z_fixed],
        [x_max, y_min, z_fixed],
    ])
    projected = []
    for corner in corners_world:
        cam_coord = np.dot(rot_mat.T, (corner - t))
        pixel = rs.rs2_project_point_to_pixel(intr, cam_coord.tolist())
        projected.append(pixel)
    return np.array(projected, dtype=np.int32)

def generate_frames(config: StreamConfig,
                    bounds_z: float = 0) -> Iterator[bytes]:
    xy_bounds = config.env_placement_bounds
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            
            # Draw bounds for each object.
            for obj_name, bounds in xy_bounds.items():
                x_min, x_max, y_min, y_max = bounds
                poly_pts = project_polygon_from_world(x_min, x_max, y_min, y_max,
                                                      bounds_z, intr, rot_mat, POS_IN_ROBOT_FRAME)
                color = (0, 255, 0)
                if obj_name == "mug":
                    color = (0, 0, 255)
                elif obj_name == "holder":
                    color = (255, 0, 0)
                cv2.polylines(frame, [poly_pts], isClosed=True, color=color, thickness=2)
                label_pos = (poly_pts[0][0], poly_pts[0][1] - 10)
                cv2.putText(frame, obj_name, label_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Blend with environment image if enabled.
            if config.blend_frames and config.env_image is not None:
                # assert size of env_image matches the frame size
                assert config.env_image.shape[0] == frame.shape[0] and config.env_image.shape[1] == frame.shape[1], \
                    f"Environment image size {config.env_image.shape} does not match frame size {frame.shape}"
                if config.blend_ratio < 1:
                    frame = cv2.addWeighted(frame, config.blend_ratio,
                                            config.env_image, 1 - config.blend_ratio, 0)
            
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')
    finally:
        pipeline.stop()

def get_env_info(env_name: str = "MugHanging", deterministic_bnds: bool = False) -> Tuple[Dict[str, Tuple[float, float, float, float]], np.ndarray]:
    # Create the environment instance with offscreen rendering.
    env = suite.make(
        env_name=env_name,
        robots="Panda_PandaUmiGripper",
        has_renderer=False,           # Use offscreen rendering.
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_widths=16 * 80,
        camera_heights=9 * 80,
        initialization_noise=None,
        camera_names=["agentview"],
    )
    env.reset()
    open_gripper_action = np.zeros(env.robots[0].dof)
    open_gripper_action[-1] = -1.0
    for _ in range(20):
        env.step(open_gripper_action)
    try:
        per_obj_bounds = env._get_initial_placement_bounds(deterministic=deterministic_bnds)
    except Exception as e:
        print(f"Error getting initial placement bounds: {e}")
        # Fallback to default bounds if the specific method fails.
        per_obj_bounds = env._get_initial_placement_bounds()
    bounds_dict = {}
    for obj_name, bounds in per_obj_bounds.items():
        x_bounds = bounds['x'] + bounds['reference'][0]
        y_bounds = bounds['y'] + bounds['reference'][1]
        bounds_dict[obj_name] = (x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1])
    
    # Retrieve the environment image from the observations.
    observations = env._get_observations(force_update=True)
    # For example, using the agent's view image.
    env_image = observations["agentview_image"][::-1]
    # env_image = observations["robot0_eye_in_hand_image"][::-1]
    # convert from float32 to uint8
    env_image = (env_image * 255).astype(np.uint8)
    print(f"Environment image shape: {env_image.shape}")
    print(f"Environment image dtype: {env_image.dtype}")
    return bounds_dict, env_image


def create_app(config: StreamConfig) -> Flask:
    app = Flask(__name__)
    
    @app.route('/video')
    def video_feed() -> Response:
        return Response(generate_frames(config),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/')
    def index() -> str:
        return '<html><body><img src="/video" width="640" height="480"></body></html>'
    
    return app

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start the RealSense Flask stream with environment blending.")
    parser.add_argument("--blend", action="store_true", help="Enable frame blending using the environment image.")
    parser.add_argument("--blend-ratio", type=float, default=0.5,
                        help="Blend ratio (e.g., 1 means a 100/0 blend between camera capture and rendering).")
    # add env name as an argument
    parser.add_argument("--env-name", type=str, default="MugHanging",
                        help="Environment name to use for blending.")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    env_image = None
    if args.blend:
        env_bounds, env_image = get_env_info(args.env_name, deterministic_bnds=False)
    print("Environment bounds:", env_bounds)
    stream_config = StreamConfig(blend_frames=args.blend, blend_ratio=args.blend_ratio, env_placement_bounds=env_bounds, env_image=env_image)
    app = create_app(stream_config)
    app.run(host='0.0.0.0', port=8012, threaded=True)
