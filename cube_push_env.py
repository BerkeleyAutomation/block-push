import os
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation

# Configure MuJoCo / OpenGL backend before importing robosuite / mujoco.
# Force EGL because OSMesa can produce intermittent black frames in this project.
if os.environ.get("MUJOCO_GL", "").lower() != "egl":
    os.environ["MUJOCO_GL"] = "egl"

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.transform_utils import convert_quat


CAMERA_NAME = "thirdview_cam"
CAMERA_POS = np.array([0.0, -1.1, 1.4])
CAMERA_TARGET = np.array([0.0, 0.0, 0.85])
CAMERA_FOVY = 45.0

TABLE_SIZE = (0.8, 0.8, 0.05)
TABLE_OFFSET = np.array([0.0, 0.0, 0.8])
TABLE_SURFACE_Z = TABLE_OFFSET[2]

# Cube half-extent and initial placement
CUBE_HALF = 0.025
# Green cube placed in front of table center (toward camera, -Y)
CUBE_POS = np.array([0.0, -0.10, TABLE_SURFACE_Z + CUBE_HALF])

# Attention camera — behind the cube on the +Y side so the gripper (approaching
# from -Y) never occludes it.  Aimed at the cube-table contact point.
ATTN_CAM_NAME   = "attn_cam"
ATTN_CAM_POS    = np.array([0.0,  0.40,  0.90])
ATTN_CAM_TARGET = np.array([0.0, -0.10,  0.80])  # cube-table contact (TABLE_SURFACE_Z=0.80)
ATTN_CAM_FOVY   = 20.0


def _look_at_quat_wxyz(pos, target):
    """Compute camera quaternion (wxyz, MuJoCo convention) for a look-at transform."""
    forward = target - pos
    forward /= np.linalg.norm(forward)

    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, up)) > 0.95:
        up = np.array([0.0, 1.0, 0.0])

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    # Camera axes in world: [right, up, -forward] (OpenGL convention: -Z looks forward)
    R_cam2world = np.column_stack([right, up, -forward])
    q_xyzw = Rotation.from_matrix(R_cam2world).as_quat()  # scipy: xyzw
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])  # wxyz


class FrankaCubePushEnv(ManipulationEnv):
    """
    Franka Panda arm pushes a single green cube from front to back on a table.
    Records a fixed third-view camera and a fixed 20×20 attention window anchored
    at the cube-table contact point.
    """

    def __init__(
        self,
        camera_height=480,
        camera_width=480,
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_object_obs=True,
        **kwargs,
    ):
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.use_object_obs = use_object_obs

        # Robosuite 1.5 uses composite controller format.
        # load_composite_controller_config flattens "arms.right" → "right"; we mirror that here.
        osc_arm_cfg = {
            "type": "OSC_POSE",
            "input_max": 1, "input_min": -1,
            "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
            "kp": 150, "damping_ratio": 1, "impedance_mode": "fixed",
            "kp_limits": [0, 300], "damping_ratio_limits": [0, 10],
            "position_limits": None, "orientation_limits": None,
            "uncouple_pos_ori": True,
            "input_type": "delta",
            "input_ref_frame": "world",
            "interpolation": None, "ramp_ratio": 0.2,
            "gripper": {"type": "GRIP"},
        }
        controller_cfg = {
            "type": "BASIC",
            # "right" must be a top-level key in body_parts (not nested under "arms")
            "body_parts": {"right": osc_arm_cfg},
        }

        super().__init__(
            robots="Panda",
            env_configuration="default",
            gripper_types="PandaGripper",
            controller_configs=controller_cfg,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            camera_names=[CAMERA_NAME, ATTN_CAM_NAME],
            camera_heights=[camera_height, camera_height],
            camera_widths=[camera_width, camera_width],
            camera_depths=False,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        super()._load_model()

        # Position robot at table edge
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](TABLE_SIZE[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        arena = TableArena(
            table_full_size=TABLE_SIZE,
            table_friction=(1.0, 5e-3, 1e-4),
            table_offset=TABLE_OFFSET,
        )
        arena.set_origin([0, 0, 0])

        # Add fixed camera via XML
        cam_quat_wxyz = _look_at_quat_wxyz(CAMERA_POS, CAMERA_TARGET)
        pos_str = " ".join(f"{v:.4f}" for v in CAMERA_POS)
        quat_str = " ".join(f"{v:.6f}" for v in cam_quat_wxyz)
        camera_elem = ET.Element("camera", attrib={
            "name": CAMERA_NAME,
            "mode": "fixed",
            "pos": pos_str,
            "quat": quat_str,
            "fovy": str(CAMERA_FOVY),
        })
        arena.worldbody.append(camera_elem)

        # Attention camera — fixed behind the cube (+Y side); gripper never obstructs it
        attn_quat_wxyz = _look_at_quat_wxyz(ATTN_CAM_POS, ATTN_CAM_TARGET)
        attn_pos_str  = " ".join(f"{v:.4f}" for v in ATTN_CAM_POS)
        attn_quat_str = " ".join(f"{v:.6f}" for v in attn_quat_wxyz)
        arena.worldbody.append(ET.Element("camera", attrib={
            "name": ATTN_CAM_NAME,
            "mode": "fixed",
            "pos":  attn_pos_str,
            "quat": attn_quat_str,
            "fovy": str(ATTN_CAM_FOVY),
        }))

        # Pale yellow cube — low friction for smooth gliding
        self.cubeG = BoxObject(
            name="cubeG",
            size_min=[CUBE_HALF, CUBE_HALF, CUBE_HALF],
            size_max=[CUBE_HALF, CUBE_HALF, CUBE_HALF],
            rgba=[1, 1, 0.6, 1],
            friction=(0.05, 0.005, 0.0001),
            density=100,
        )

        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.cubeG],
        )

    # ------------------------------------------------------------------
    # References + reset
    # ------------------------------------------------------------------

    def _setup_references(self):
        super()._setup_references()
        self.cubeG_body_id = self.sim.model.body_name2id(self.cubeG.root_body)

    def _reset_internal(self):
        super()._reset_internal()
        # Place cube at fixed deterministic position
        jname = self.cubeG.joints[0]
        quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0])  # no rotation
        self.sim.data.set_joint_qpos(
            jname, np.concatenate([CUBE_POS, quat_xyzw])
        )
        self.sim.forward()
 # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------

    def _setup_observables(self):
        observables = super()._setup_observables()
        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def cubeG_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cubeG_body_id])

            @sensor(modality=modality)
            def cubeG_quat(obs_cache):
                return convert_quat(
                    np.array(self.sim.data.body_xquat[self.cubeG_body_id]), to="xyzw"
                )

            for s in [cubeG_pos, cubeG_quat]:
                observables[s.__name__] = Observable(
                    name=s.__name__, sensor=s, sampling_rate=self.control_freq
                )
        return observables

    def reward(self, action):
        return 0.0

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def get_cube_pos(self):
        """World-space position of the green cube."""
        return np.array(self.sim.data.body_xpos[self.cubeG_body_id])

    def get_camera_extrinsics(self):
        """
        Returns (cam_pos, cam_R) where cam_R is a 3×3 matrix whose columns are
        the camera-frame axes expressed in world coordinates.
        (Camera looks in the -Z direction of this frame.)
        """
        cam_id = self.sim.model.camera_name2id(CAMERA_NAME)
        cam_pos = self.sim.model.cam_pos[cam_id].copy()
        wxyz = self.sim.model.cam_quat[cam_id].copy()
        # Convert wxyz → xyzw for scipy
        q_xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
        cam_R = Rotation.from_quat(q_xyzw).as_matrix()
        return cam_pos, cam_R

    def get_focal_length(self, fovy=CAMERA_FOVY):
        """Focal length in pixels for the thirdview camera."""
        return (self.camera_height / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)

    def get_camera_params(self):
        """Return (cam_pos, cam_R, f, W, H) for the thirdview camera.

        Convenience bundle for servoing math; avoids multiple calls.
        """
        cam_pos, cam_R = self.get_camera_extrinsics()
        f = self.get_focal_length()
        return cam_pos, cam_R, f, self.camera_width, self.camera_height

    def get_attn_camera_params(self):
        """Return (cam_pos, cam_R, f, W, H) for the attention camera."""
        cam_id = self.sim.model.camera_name2id(ATTN_CAM_NAME)
        cam_pos = self.sim.model.cam_pos[cam_id].copy()
        wxyz = self.sim.model.cam_quat[cam_id].copy()
        q_xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
        cam_R = Rotation.from_quat(q_xyzw).as_matrix()
        f = (self.camera_height / 2.0) / np.tan(np.deg2rad(ATTN_CAM_FOVY) / 2.0)
        return cam_pos, cam_R, f, self.camera_width, self.camera_height

    def world_to_pixel(self, point_3d, fovy=CAMERA_FOVY):
        """
        Project a 3D world point to (u, v) pixel coordinates in the thirdview camera.
        Accounts for robosuite's top-down image convention (OpenGL origin at bottom-left).
        """
        W, H = self.camera_width, self.camera_height
        cam_pos, cam_R = self.get_camera_extrinsics()

        # Into camera frame (camera looks in -Z)
        p_cam = cam_R.T @ (point_3d - cam_pos)

        f = (H / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
        # Standard OpenGL pinhole projection.  Camera looks in -Z, +Y is up.
        # OpenGL: u_gl = W/2 + X*f/(-Z); v_gl = H/2 + Y*f/(-Z).
        u_gl = -f * p_cam[0] / p_cam[2] + W / 2.0
        v_gl = -f * p_cam[1] / p_cam[2] + H / 2.0
        u = int(round(u_gl))
        # OpenGL origin is bottom-left; get_frame flips vertically to screen-top-left,
        # so we must mirror v to match the flipped frame.
        v = int(round(H - 1 - v_gl))
        return u, v

    def attn_world_to_pixel(self, point_3d):
        """Project a 3D world point to (u, v) pixel coords in the attention camera."""
        W, H = self.camera_width, self.camera_height
        cam_id = self.sim.model.camera_name2id(ATTN_CAM_NAME)
        cam_pos = self.sim.model.cam_pos[cam_id].copy()
        wxyz = self.sim.model.cam_quat[cam_id].copy()
        q_xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
        cam_R = Rotation.from_quat(q_xyzw).as_matrix()
        p_cam = cam_R.T @ (point_3d - cam_pos)
        f = (H / 2.0) / np.tan(np.deg2rad(ATTN_CAM_FOVY) / 2.0)
        u_gl = -f * p_cam[0] / p_cam[2] + W / 2.0
        v_gl = -f * p_cam[1] / p_cam[2] + H / 2.0
        u = int(round(u_gl))
        v = int(round(H - 1 - v_gl))
        return u, v

    def get_attention_frame(self, obs):
        """RGB frame from the attention camera (behind cube, gripper-free view)."""
        frame = obs[f"{ATTN_CAM_NAME}_image"]
        return frame[::-1].copy()

    def get_frame(self, obs):
        """Return the current (H, W, 3) uint8 RGB camera frame from observations."""
        frame = obs[f"{CAMERA_NAME}_image"]
        # Robosuite returns images as uint8; flip vertically (OpenGL → top-down)
        return frame[::-1].copy()
      
