# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import repair_episodes_metadata_if_missing, validate_dataset_for_resume
from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    AddTeleopActionAsComplimentaryDataStep,
    AddTeleopEventsAsInfoStep,
    DataProcessorPipeline,
    DeviceProcessorStep,
    EnvTransition,
    GripperPenaltyProcessorStep,
    ImageCropResizeProcessorStep,
    InterventionActionProcessorStep,
    JointVelocityProcessorStep,
    LeaderInterventionActionProcessorStep,
    ApplyEEPolicyActionPipelineStep,
    MapDeltaActionToRobotActionStep,
    MapTensorToDeltaActionDictStep,
    MotorCurrentProcessorStep,
    Numpy2TorchActionProcessorStep,
    RewardClassifierProcessorStep,
    RobotActionToPolicyActionProcessorStep,
    TimeLimitProcessorStep,
    Torch2NumpyActionProcessorStep,
    TransitionKey,
    VanillaObservationProcessorStep,
    create_transition,
)
from lerobot.processor.converters import identity_transition
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.robots.robot import Robot
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    ClipJointPositions,
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    ForwardKinematicsJointsToEEObservation,
    GripperVelocityToJoint,
    InverseKinematicsRLStep,
)
from lerobot.teleoperators import (
    gamepad,  # noqa: F401
    keyboard,  # noqa: F401
    make_teleoperator_from_config,
    so101_leader,  # noqa: F401
)
from lerobot.teleoperators.so101_leader.so101_leader_hil import (
    as_hil_leader_teleop,
    maybe_align_leader_for_intervention,
    maybe_hold_leader_teleop,
    maybe_mirror_leader_teleop,
    maybe_release_leader_for_teleop,
    maybe_reset_leader_teleop,
)
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGES, OBS_STATE, REWARD
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

logging.basicConfig(level=logging.INFO)


def _ee_dataset_action_features(use_gripper: bool) -> dict[str, Any]:
    """LeRobot feature schema for HIL-SERL end-effector delta actions."""
    names: dict[str, int] = {"delta_x": 0, "delta_y": 1, "delta_z": 2}
    if use_gripper:
        names["gripper"] = 3
    return {"dtype": "float32", "shape": (len(names),), "names": names}


def _resolve_record_action_features(action_features: Any, use_gripper: bool) -> dict[str, Any]:
    """Normalize teleop.action_features into a LeRobot dataset feature dict.

    Keyboard/gamepad teleops already return `{dtype, shape, names}`. Leader arms return a
    motor map `{motor.pos: float}` which is invalid for `LeRobotDataset.create`.
    """
    if isinstance(action_features, dict) and "dtype" in action_features:
        return action_features
    return _ee_dataset_action_features(use_gripper)


def _print_start_recording(episode_idx: int, num_episodes: int) -> None:
    """Clear on-screen cue that teleop frames are about to be saved."""
    msg = f">>> START RECORDING  episode {episode_idx + 1}/{num_episodes}  — move the leader now (s=成功 q=失败 r=重录)"
    print()
    print("=" * 72)
    print(msg)
    print("=" * 72)
    print(flush=True)
    logging.info(msg)


def _to_rerun_value(value: Any) -> Any:
    """Convert torch/numpy observation values into rerun-friendly arrays."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim >= 1 and value.shape[0] == 1:
            value = value.squeeze(0)
        value = value.numpy()
    if isinstance(value, np.ndarray) and value.ndim == 3:
        if value.shape[0] in (1, 3, 4) and value.shape[-1] not in (1, 3, 4):
            value = np.transpose(value, (1, 2, 0))
        if value.dtype != np.uint8:
            value = (
                (np.clip(value, 0.0, 1.0) * 255).astype(np.uint8)
                if float(np.nanmax(value)) <= 1.5
                else np.clip(value, 0, 255).astype(np.uint8)
            )
    return value


def _log_transition_to_rerun(
    transition: EnvTransition, env: gym.Env | None = None, *, step: int | None = None
) -> None:
    """Stream cameras / state / action into the Rerun viewer.

    Cameras are taken from the raw robot feed (e.g. 640x480) so the viewer is sharp.
    Dataset recording still uses the resized 128x128 pipeline images.
    """
    obs_rr: dict[str, Any] = {}

    # Prefer full-resolution camera frames for display (cached from last env.step).
    raw = getattr(env, "_last_observation", None) if env is not None else None
    if raw is None and env is not None and hasattr(env, "_get_observation"):
        raw = env._get_observation()
    pixels = raw.get("pixels") if isinstance(raw, dict) else None
    if isinstance(pixels, dict):
        for cam_key, image in pixels.items():
            if image is not None:
                obs_rr[f"cameras.{cam_key}"] = _to_rerun_value(image)

    # Fall back to processed images only if raw cameras are unavailable.
    if not any(k.startswith("cameras.") for k in obs_rr):
        observation = transition.get(TransitionKey.OBSERVATION, {}) or {}
        for key, value in observation.items():
            if value is not None and "image" in str(key):
                obs_rr[key] = _to_rerun_value(value)

    observation = transition.get(TransitionKey.OBSERVATION, {}) or {}
    if OBS_STATE in observation and observation[OBS_STATE] is not None:
        obs_rr[OBS_STATE] = _to_rerun_value(observation[OBS_STATE])

    action = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(
        "teleop_action", transition.get(TransitionKey.ACTION)
    )
    action_rr: dict[str, Any] = {}
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().reshape(-1).numpy()
    if isinstance(action, np.ndarray) and action.ndim == 1:
        names = ["delta_x", "delta_y", "delta_z", "gripper"]
        for i, val in enumerate(action.tolist()):
            action_rr[names[i] if i < len(names) else str(i)] = float(val)
    elif isinstance(action, dict):
        action_rr = {
            key: float(val)
            for key, val in action.items()
            if isinstance(val, (int, float, np.floating, np.integer))
        }

    log_rerun_data(observation=obs_rr, action=action_rr or None, step=step)


def _action_tensor_for_record(action_to_record: Any, use_gripper: bool) -> torch.Tensor:
    """Coerce teleop/policy actions into a 1D float tensor for dataset frames."""
    if isinstance(action_to_record, torch.Tensor):
        return action_to_record.detach().cpu().reshape(-1).float()
    if isinstance(action_to_record, np.ndarray):
        return torch.from_numpy(np.asarray(action_to_record)).reshape(-1).float()
    if isinstance(action_to_record, dict) and {"delta_x", "delta_y", "delta_z"}.issubset(action_to_record):
        values = [
            float(action_to_record["delta_x"]),
            float(action_to_record["delta_y"]),
            float(action_to_record["delta_z"]),
        ]
        if use_gripper:
            values.append(float(action_to_record.get("gripper", 1.0)))
        return torch.tensor(values, dtype=torch.float32)
    # Leader sometimes leaves raw joint dicts; hold still in EE space rather than crash.
    if isinstance(action_to_record, dict) and any(
        isinstance(k, str) and k.endswith(".pos") for k in action_to_record
    ):
        values = [0.0, 0.0, 0.0]
        if use_gripper:
            values.append(1.0)
        return torch.tensor(values, dtype=torch.float32)
    raise TypeError(
        f"Cannot record action of type {type(action_to_record).__name__}; "
        "expected EE delta tensor/dict from teleop."
    )


@dataclass
class DatasetConfig:
    """Configuration for dataset creation and management."""

    repo_id: str
    task: str
    root: str | None = None
    num_episodes_to_record: int = 5
    replay_episode: int | None = None
    push_to_hub: bool = False
    resume: bool = False


@dataclass
class GymManipulatorConfig:
    """Main configuration for gym manipulator environment."""

    env: HILSerlRobotEnvConfig
    dataset: DatasetConfig
    mode: str | None = None  # Either "record", "replay", None
    device: str = "cpu"


def reset_follower_position(robot_arm: Robot, target_position: np.ndarray) -> None:
    """Reset robot arm to target position using smooth trajectory."""
    current_position_dict = robot_arm.bus.sync_read("Present_Position")
    current_position = np.array(
        [current_position_dict[name] for name in current_position_dict], dtype=np.float32
    )
    trajectory = torch.from_numpy(
        np.linspace(current_position, target_position, 50)
    )  # NOTE: 30 is just an arbitrary number
    for pose in trajectory:
        action_dict = dict(zip(current_position_dict, pose, strict=False))
        robot_arm.bus.sync_write("Goal_Position", action_dict)
        precise_sleep(0.015)


class RobotEnv(gym.Env):
    """Gym environment for robotic control with human intervention support."""

    def __init__(
        self,
        robot,
        use_gripper: bool = False,
        display_cameras: bool = False,
        reset_pose: list[float] | None = None,
        reset_time_s: float = 5.0,
    ) -> None:
        """Initialize robot environment with configuration options.

        Args:
            robot: Robot interface for hardware communication.
            use_gripper: Whether to include gripper in action space.
            display_cameras: Whether to show camera feeds during execution.
            reset_pose: Joint positions for environment reset.
            reset_time_s: Time to wait during reset.
        """
        super().__init__()

        self.robot = robot
        self.display_cameras = display_cameras

        # Connect to the robot if not already connected.
        if not self.robot.is_connected:
            self.robot.connect()

        # Episode tracking.
        self.current_step = 0
        self.episode_data = None

        self._joint_names = [f"{key}.pos" for key in self.robot.bus.motors]
        self._image_keys = self.robot.cameras.keys()

        self.reset_pose = reset_pose
        self.reset_time_s = reset_time_s

        self.use_gripper = use_gripper

        self._joint_names = list(self.robot.bus.motors.keys())
        self._raw_joint_positions = None

        self._setup_spaces()

    def _get_observation(self) -> dict[str, Any]:
        """Get current robot observation including joint positions and camera images."""
        obs_dict = self.robot.get_observation()
        raw_joint_joint_position = {f"{name}.pos": obs_dict[f"{name}.pos"] for name in self._joint_names}
        joint_positions = np.array([raw_joint_joint_position[f"{name}.pos"] for name in self._joint_names])

        images = {key: obs_dict[key] for key in self._image_keys}

        return {"agent_pos": joint_positions, "pixels": images, **raw_joint_joint_position}

    def _setup_spaces(self) -> None:
        """Configure observation and action spaces based on robot capabilities."""
        current_observation = self._get_observation()

        observation_spaces = {}

        # Define observation spaces for images and other states.
        if current_observation is not None and "pixels" in current_observation:
            prefix = OBS_IMAGES
            observation_spaces = {
                f"{prefix}.{key}": gym.spaces.Box(
                    low=0, high=255, shape=current_observation["pixels"][key].shape, dtype=np.uint8
                )
                for key in current_observation["pixels"]
            }

        if current_observation is not None:
            agent_pos = current_observation["agent_pos"]
            observation_spaces[OBS_STATE] = gym.spaces.Box(
                low=0,
                high=10,
                shape=agent_pos.shape,
                dtype=np.float32,
            )

        self.observation_space = gym.spaces.Dict(observation_spaces)

        # Define the action space for joint positions along with setting an intervention flag.
        action_dim = 3
        bounds = {}
        bounds["min"] = -np.ones(action_dim)
        bounds["max"] = np.ones(action_dim)

        if self.use_gripper:
            action_dim += 1
            bounds["min"] = np.concatenate([bounds["min"], [0]])
            bounds["max"] = np.concatenate([bounds["max"], [2]])

        self.action_space = gym.spaces.Box(
            low=bounds["min"],
            high=bounds["max"],
            shape=(action_dim,),
            dtype=np.float32,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options.

        Returns:
            Tuple of (observation, info) dictionaries.
        """
        # Reset the robot to the configured home pose (fixed_reset_joint_positions).
        start_time = time.perf_counter()
        if self.reset_pose is not None:
            print(f"Resetting environment → home pose {list(self.reset_pose)} ({self.reset_time_s:.1f}s)")
            logging.info("Resetting environment to %s", self.reset_pose)
            log_say("Reset the environment.", play_sounds=True)
            reset_follower_position(self.robot, np.array(self.reset_pose))
            log_say("Reset the environment done.", play_sounds=True)
            print("Resetting environment done.")
        else:
            logging.warning("No fixed_reset_joint_positions; skipping arm reset")

        precise_sleep(self.reset_time_s - (time.perf_counter() - start_time))

        super().reset(seed=seed, options=options)

        # Reset episode tracking variables.
        self.current_step = 0
        self.episode_data = None
        obs = self._get_observation()
        self._raw_joint_positions = {f"{key}.pos": obs[f"{key}.pos"] for key in self._joint_names}
        return obs, {TeleopEvents.IS_INTERVENTION: False}

    def step(self, action) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute one environment step with given action."""
        joint_targets_dict = {f"{key}.pos": action[i] for i, key in enumerate(self.robot.bus.motors.keys())}

        self.robot.send_action(joint_targets_dict)

        obs = self._get_observation()
        self._last_observation = obs

        self._raw_joint_positions = {f"{key}.pos": obs[f"{key}.pos"] for key in self._joint_names}

        if self.display_cameras:
            self.render(obs)

        self.current_step += 1

        reward = 0.0
        terminated = False
        truncated = False

        return (
            obs,
            reward,
            terminated,
            truncated,
            {TeleopEvents.IS_INTERVENTION: False},
        )

    def render(self, observation: dict[str, Any] | None = None) -> None:
        """Display robot camera feeds from the pixels dict (front/wrist)."""
        import cv2

        # opencv-python-headless has no highgui; disable display after first failure.
        if getattr(self, "_display_unavailable", False):
            return

        current_observation = observation if observation is not None else self._get_observation()
        if not current_observation:
            return

        # RobotEnv stores cameras under "pixels": {"front": HWC, "wrist": HWC}.
        pixels = current_observation.get("pixels")
        if not isinstance(pixels, dict) or not pixels:
            pixels = {
                key: current_observation[key]
                for key in current_observation
                if isinstance(key, str) and ("image" in key or key in self._image_keys)
            }
        if not pixels:
            return

        try:
            for key, image in pixels.items():
                if image is None:
                    continue
                if hasattr(image, "numpy"):
                    image = image.detach().cpu().numpy()
                image = np.asarray(image)
                if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
                    image = np.transpose(image, (1, 2, 0))  # CHW -> HWC
                if image.dtype != np.uint8:
                    image = (
                        (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
                        if float(np.nanmax(image)) <= 1.5
                        else image.astype(np.uint8)
                    )
                bgr = (
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    if image.ndim == 3 and image.shape[-1] == 3
                    else image
                )
                cv2.imshow(f"camera.{key}", bgr)
            cv2.waitKey(1)
        except cv2.error as exc:
            self._display_unavailable = True
            logging.warning(
                "Camera display unavailable (%s). Install GUI OpenCV: "
                "pip uninstall -y opencv-python-headless && pip install opencv-python",
                exc,
            )

    def close(self) -> None:
        """Close environment and disconnect robot."""
        if self.robot.is_connected:
            self.robot.disconnect()

    def get_raw_joint_positions(self) -> dict[str, float]:
        """Get raw joint positions."""
        return self._raw_joint_positions


def make_robot_env(cfg: HILSerlRobotEnvConfig) -> tuple[gym.Env, Any]:
    """Create robot environment from configuration.

    Args:
        cfg: Environment configuration.

    Returns:
        Tuple of (gym environment, teleoperator device).
    """
    # Check if this is a GymHIL simulation environment
    if cfg.name == "gym_hil":
        assert cfg.robot is None and cfg.teleop is None, "GymHIL environment does not support robot or teleop"
        import gym_hil  # noqa: F401

        # Extract gripper settings with defaults
        use_gripper = cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else True
        gripper_penalty = cfg.processor.gripper.gripper_penalty if cfg.processor.gripper is not None else 0.0

        env = gym.make(
            f"gym_hil/{cfg.task}",
            image_obs=True,
            render_mode="human",
            use_gripper=use_gripper,
            gripper_penalty=gripper_penalty,
        )

        return env, None

    # Real robot environment
    assert cfg.robot is not None, "Robot config must be provided for real robot environment"
    assert cfg.teleop is not None, "Teleop config must be provided for real robot environment"

    robot = make_robot_from_config(cfg.robot)
    teleop_device = make_teleoperator_from_config(cfg.teleop)
    if cfg.processor.control_mode == "leader":
        teleop_device = as_hil_leader_teleop(
            teleop_device,
            mirror_leader_on_follower=cfg.processor.mirror_leader_on_follower,
        )
    teleop_device.connect()

    # Create base environment with safe defaults
    use_gripper = cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else True
    display_cameras = (
        cfg.processor.observation.display_cameras if cfg.processor.observation is not None else False
    )
    reset_pose = cfg.processor.reset.fixed_reset_joint_positions if cfg.processor.reset is not None else None
    reset_time_s = cfg.processor.reset.reset_time_s if cfg.processor.reset is not None else 5.0

    env = RobotEnv(
        robot=robot,
        use_gripper=use_gripper,
        display_cameras=display_cameras,
        reset_pose=reset_pose,
        reset_time_s=reset_time_s,
    )

    return env, teleop_device


def make_processors(
    env: gym.Env, teleop_device: Teleoperator | None, cfg: HILSerlRobotEnvConfig, device: str = "cpu"
) -> tuple[
    DataProcessorPipeline[EnvTransition, EnvTransition], DataProcessorPipeline[EnvTransition, EnvTransition]
]:
    """Create environment and action processors.

    Args:
        env: Robot environment instance.
        teleop_device: Teleoperator device for intervention.
        cfg: Processor configuration.
        device: Target device for computations.

    Returns:
        Tuple of (environment processor, action processor).
    """
    terminate_on_success = (
        cfg.processor.reset.terminate_on_success if cfg.processor.reset is not None else True
    )

    if cfg.name == "gym_hil":
        action_pipeline_steps = [
            InterventionActionProcessorStep(terminate_on_success=terminate_on_success),
            Torch2NumpyActionProcessorStep(),
        ]

        env_pipeline_steps = [
            Numpy2TorchActionProcessorStep(),
            VanillaObservationProcessorStep(),
            AddBatchDimensionProcessorStep(),
            DeviceProcessorStep(device=device),
        ]

        return DataProcessorPipeline(
            steps=env_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
        ), DataProcessorPipeline(
            steps=action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
        )

    # Full processor pipeline for real robot environment
    # Get robot and motor information for kinematics
    motor_names = list(env.robot.bus.motors.keys())

    # Set up kinematics solver if inverse kinematics is configured
    kinematics_solver = None
    if cfg.processor.inverse_kinematics is not None:
        kinematics_solver = RobotKinematics(
            urdf_path=cfg.processor.inverse_kinematics.urdf_path,
            target_frame_name=cfg.processor.inverse_kinematics.target_frame_name,
            joint_names=motor_names,
        )

    env_pipeline_steps = [VanillaObservationProcessorStep()]

    if cfg.processor.observation is not None:
        if cfg.processor.observation.add_joint_velocity_to_observation:
            env_pipeline_steps.append(JointVelocityProcessorStep(dt=1.0 / cfg.fps))
        if cfg.processor.observation.add_current_to_observation:
            env_pipeline_steps.append(MotorCurrentProcessorStep(robot=env.robot))

    if kinematics_solver is not None:
        env_pipeline_steps.append(
            ForwardKinematicsJointsToEEObservation(
                kinematics=kinematics_solver,
                motor_names=motor_names,
            )
        )

    if cfg.processor.image_preprocessing is not None:
        env_pipeline_steps.append(
            ImageCropResizeProcessorStep(
                crop_params_dict=cfg.processor.image_preprocessing.crop_params_dict,
                resize_size=cfg.processor.image_preprocessing.resize_size,
            )
        )

    # Add time limit processor if reset config exists
    if cfg.processor.reset is not None:
        env_pipeline_steps.append(
            TimeLimitProcessorStep(max_episode_steps=int(cfg.processor.reset.control_time_s * cfg.fps))
        )

    # Add gripper penalty processor if gripper config exists and enabled
    if cfg.processor.gripper is not None and cfg.processor.gripper.use_gripper:
        env_pipeline_steps.append(
            GripperPenaltyProcessorStep(
                penalty=cfg.processor.gripper.gripper_penalty,
                max_gripper_pos=cfg.processor.max_gripper_pos,
            )
        )

    if (
        cfg.processor.reward_classifier is not None
        and cfg.processor.reward_classifier.pretrained_path is not None
    ):
        env_pipeline_steps.append(
            RewardClassifierProcessorStep(
                pretrained_path=cfg.processor.reward_classifier.pretrained_path,
                device=device,
                success_threshold=cfg.processor.reward_classifier.success_threshold,
                success_reward=cfg.processor.reward_classifier.success_reward,
                terminate_on_success=(
                    cfg.processor.reward_classifier.terminate_on_success
                    if cfg.processor.reward_classifier.terminate_on_success is not None
                    else terminate_on_success
                ),
            )
        )

    env_pipeline_steps.append(AddBatchDimensionProcessorStep())
    env_pipeline_steps.append(DeviceProcessorStep(device=device))

    use_gripper = cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else False
    if cfg.processor.control_mode == "leader":
        ee_step_sizes = (
            cfg.processor.inverse_kinematics.end_effector_step_sizes
            if cfg.processor.inverse_kinematics is not None
            else None
        )
        intervention_step = LeaderInterventionActionProcessorStep(
            motor_names=motor_names,
            use_gripper=use_gripper,
            terminate_on_success=terminate_on_success,
            freeze_policy_without_intervention=cfg.processor.freeze_policy_without_intervention,
            kinematics=kinematics_solver,
            end_effector_step_sizes=ee_step_sizes,
        )
    else:
        intervention_step = InterventionActionProcessorStep(
            use_gripper=use_gripper,
            terminate_on_success=terminate_on_success,
            freeze_policy_without_intervention=cfg.processor.freeze_policy_without_intervention,
        )

    action_pipeline_steps = [
        AddTeleopActionAsComplimentaryDataStep(teleop_device=teleop_device),
        AddTeleopEventsAsInfoStep(teleop_device=teleop_device),
        intervention_step,
    ]

    # Replace InverseKinematicsProcessor with new kinematic processors
    if cfg.processor.inverse_kinematics is not None and kinematics_solver is not None:
        ee_action_pipeline_steps = [
            MapTensorToDeltaActionDictStep(
                use_gripper=use_gripper,
            ),
            MapDeltaActionToRobotActionStep(),
            EEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes=cfg.processor.inverse_kinematics.end_effector_step_sizes,
                motor_names=motor_names,
                use_latched_reference=False,
                use_ik_solution=True,
            ),
            EEBoundsAndSafety(
                end_effector_bounds=cfg.processor.inverse_kinematics.end_effector_bounds,
                max_ee_step_m=cfg.processor.inverse_kinematics.max_ee_step_m,
            ),
            GripperVelocityToJoint(
                clip_max=cfg.processor.max_gripper_pos,
                speed_factor=1.0,
                discrete_gripper=True,
            ),
            InverseKinematicsRLStep(
                kinematics=kinematics_solver, motor_names=motor_names, initial_guess_current_joints=False
            ),
        ]
        ee_action_pipeline = DataProcessorPipeline(
            steps=ee_action_pipeline_steps,
            to_transition=identity_transition,
            to_output=identity_transition,
        )
        if cfg.processor.control_mode == "leader":
            action_pipeline_steps.append(
                ApplyEEPolicyActionPipelineStep(ee_action_pipeline=ee_action_pipeline)
            )
        else:
            action_pipeline_steps.extend(ee_action_pipeline_steps)
        if cfg.processor.inverse_kinematics.joint_position_bounds:
            action_pipeline_steps.append(
                ClipJointPositions(
                    joint_position_bounds=cfg.processor.inverse_kinematics.joint_position_bounds
                )
            )
        action_pipeline_steps.append(RobotActionToPolicyActionProcessorStep(motor_names=motor_names))

    return DataProcessorPipeline(
        steps=env_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
    ), DataProcessorPipeline(
        steps=action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
    )


def step_env_and_process_transition(
    env: gym.Env,
    transition: EnvTransition,
    action: torch.Tensor,
    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    teleop_device: Teleoperator | None = None,
) -> EnvTransition:
    """
    Execute one step with processor pipeline.

    Args:
        env: The robot environment
        transition: Current transition state
        action: Action to execute
        env_processor: Environment processor
        action_processor: Action processor

    Returns:
        Processed transition with updated state.
    """

    # If Space requested intervention, align leader→follower before takeover so the
    # follower does not jump to the leader's idle/home pose.
    if teleop_device is not None and hasattr(env, "get_raw_joint_positions"):
        maybe_align_leader_for_intervention(teleop_device, env.get_raw_joint_positions())

    # Create action transition
    transition[TransitionKey.ACTION] = action
    transition[TransitionKey.OBSERVATION] = (
        env.get_raw_joint_positions() if hasattr(env, "get_raw_joint_positions") else {}
    )
    processed_action_transition = action_processor(transition)
    processed_action = processed_action_transition[TransitionKey.ACTION]

    obs, reward, terminated, truncated, info = env.step(processed_action)

    reward = reward + processed_action_transition[TransitionKey.REWARD]
    terminated = terminated or processed_action_transition[TransitionKey.DONE]
    truncated = truncated or processed_action_transition[TransitionKey.TRUNCATED]
    complementary_data = processed_action_transition[TransitionKey.COMPLEMENTARY_DATA].copy()
    new_info = processed_action_transition[TransitionKey.INFO].copy()
    new_info.update(info)

    new_transition = create_transition(
        observation=obs,
        action=processed_action,
        reward=reward,
        done=terminated,
        truncated=truncated,
        info=new_info,
        complementary_data=complementary_data,
    )
    new_transition = env_processor(new_transition)

    if teleop_device is not None and hasattr(env, "get_raw_joint_positions"):
        maybe_mirror_leader_teleop(teleop_device, env.get_raw_joint_positions())

    return new_transition


def control_loop(
    env: gym.Env,
    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    teleop_device: Teleoperator,
    cfg: GymManipulatorConfig,
) -> None:
    """Main control loop for robot environment interaction.
    if cfg.mode == "record": then a dataset will be created and recorded

    Args:
     env: The robot environment
     env_processor: Environment processor
     action_processor: Action processor
     teleop_device: Teleoperator device
     cfg: gym_manipulator configuration
    """
    dt = 1.0 / cfg.env.fps
    use_rerun = bool(
        cfg.env.processor.observation is not None and cfg.env.processor.observation.display_cameras
    )
    if use_rerun:
        # Prefer Rerun viewer over OpenCV imshow. Init before the control loop so
        # restarts can reconnect to an already-open viewer.
        if hasattr(env, "display_cameras"):
            env.display_cameras = False
        print("Starting Rerun viewer (display_cameras=true)...")
        init_rerun(session_name="hilserl_record")
        print("Rerun viewer ready — watch observation.cameras.front / .wrist (full 640x480)")
        print("Tip: leave this window open; next run will reconnect instead of respawning")

    print(f"Starting control loop at {cfg.env.fps} FPS")
    if cfg.env.processor.control_mode == "leader":
        print("Controls (leader arm):")
        print("- Space: toggle human intervention")
        print("  (first aligns leader→follower, then you drag; follower follows leader)")
        print("- s: success | q: fail | r: rerecord")
        print("- Note: s/q ends the episode and resets FOLLOWER to home (not Space)")
        if cfg.mode == "record":
            print("- After s/q: both arms → home (leader holds), then 2s to grab, then free teleop")
        elif cfg.env.processor.mirror_leader_on_follower:
            print("- When not intervening, leader mirrors follower")
        else:
            print("- When not intervening, follower runs policy; leader arm stays free")
    else:
        print("Controls:")
        print("- Use gamepad/teleop device for intervention")
        print("- When not intervening, robot will stay still")
    print("- Press Ctrl+C to exit")

    # Reset environment and processors
    obs, info = env.reset()
    complementary_data = (
        {"raw_joint_positions": info.pop("raw_joint_positions")} if "raw_joint_positions" in info else {}
    )
    env_processor.reset()
    action_processor.reset()

    # Process initial observation
    transition = create_transition(observation=obs, info=info, complementary_data=complementary_data)
    transition = env_processor(data=transition)
    if cfg.env.processor.control_mode == "leader" and cfg.mode == "record":
        set_intervention = getattr(teleop_device, "_set_intervention", None)
        if callable(set_intervention):
            # Keep torque while parking at home; release after grab settle.
            set_intervention(True, release_torque=False)
        maybe_reset_leader_teleop(teleop_device, env.get_raw_joint_positions())
        maybe_release_leader_for_teleop(teleop_device, settle_s=2.0)

    # Determine if gripper is used
    use_gripper = cfg.env.processor.gripper.use_gripper if cfg.env.processor.gripper is not None else True

    dataset = None
    episode_idx = 0
    try:
        if cfg.mode == "record":
            logging.info(
                "Starting recording: %d episodes → %s",
                cfg.dataset.num_episodes_to_record,
                cfg.dataset.root or cfg.dataset.repo_id,
            )
            action_features = _resolve_record_action_features(
                teleop_device.action_features if teleop_device is not None else None,
                use_gripper,
            )
            features = {
                ACTION: action_features,
                REWARD: {"dtype": "float32", "shape": (1,), "names": None},
                DONE: {"dtype": "bool", "shape": (1,), "names": None},
            }
            if use_gripper:
                features["complementary_info.discrete_penalty"] = {
                    "dtype": "float32",
                    "shape": (1,),
                    "names": ["discrete_penalty"],
                }

            for key, value in transition[TransitionKey.OBSERVATION].items():
                if key == OBS_STATE:
                    features[key] = {
                        "dtype": "float32",
                        "shape": value.squeeze(0).shape,
                        "names": None,
                    }
                if "image" in key:
                    features[key] = {
                        "dtype": "video",
                        "shape": value.squeeze(0).shape,
                        "names": ["channels", "height", "width"],
                    }

            dataset_root = Path(cfg.dataset.root) if cfg.dataset.root else None
            can_resume = (
                cfg.dataset.resume
                and dataset_root is not None
                and (dataset_root / "meta/info.json").exists()
            )
            if can_resume:
                is_valid, message = validate_dataset_for_resume(dataset_root)
                if not is_valid:
                    raise ValueError(f"Cannot resume dataset at {cfg.dataset.root}: {message}")
                if repair_episodes_metadata_if_missing(dataset_root):
                    logging.info("Rebuilt missing meta/episodes from data parquet")
                dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root)
                dataset.start_image_writer(num_processes=0, num_threads=4)
                dataset.meta.metadata_buffer_size = 1
                episode_idx = dataset.meta.total_episodes
                logging.info(
                    "Resuming dataset recording at %s (%d/%d episodes)",
                    cfg.dataset.root,
                    episode_idx,
                    cfg.dataset.num_episodes_to_record,
                )
            else:
                dataset = LeRobotDataset.create(
                    cfg.dataset.repo_id,
                    cfg.env.fps,
                    root=cfg.dataset.root,
                    use_videos=True,
                    image_writer_threads=4,
                    image_writer_processes=0,
                    features=features,
                )
                dataset.meta.metadata_buffer_size = 1
                episode_idx = 0

        episode_step = 0
        episode_start_time = time.perf_counter()
        if cfg.mode == "record":
            _print_start_recording(episode_idx, cfg.dataset.num_episodes_to_record)

        while episode_idx < cfg.dataset.num_episodes_to_record:
            step_start_time = time.perf_counter()

            # Create a neutral action (no movement)
            neutral_action = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
            if use_gripper:
                neutral_action = torch.cat([neutral_action, torch.tensor([1.0])])  # Gripper stay

            # Use the new step function
            transition = step_env_and_process_transition(
                env=env,
                transition=transition,
                action=neutral_action,
                env_processor=env_processor,
                action_processor=action_processor,
                teleop_device=teleop_device,
            )
            terminated = transition.get(TransitionKey.DONE, False)
            truncated = transition.get(TransitionKey.TRUNCATED, False)

            if use_rerun:
                _log_transition_to_rerun(transition, env=env)

            if cfg.mode == "record":
                observations = {
                    k: v.squeeze(0).cpu()
                    for k, v in transition[TransitionKey.OBSERVATION].items()
                    if isinstance(v, torch.Tensor)
                }
                # Prefer teleop_action (EE deltas for both keyboard and leader).
                action_to_record = transition[TransitionKey.COMPLEMENTARY_DATA].get(
                    "teleop_action", transition[TransitionKey.ACTION]
                )
                frame = {
                    **observations,
                    ACTION: _action_tensor_for_record(action_to_record, use_gripper),
                    REWARD: np.array([transition[TransitionKey.REWARD]], dtype=np.float32),
                    DONE: np.array([terminated or truncated], dtype=bool),
                }
                if use_gripper:
                    discrete_penalty = transition[TransitionKey.COMPLEMENTARY_DATA].get(
                        "discrete_penalty", 0.0
                    )
                    frame["complementary_info.discrete_penalty"] = np.array(
                        [discrete_penalty], dtype=np.float32
                    )

                if dataset is not None:
                    frame["task"] = cfg.dataset.task
                    dataset.add_frame(frame)

            episode_step += 1

            # Handle episode termination
            if terminated or truncated:
                episode_time = time.perf_counter() - episode_start_time
                logging.info(
                    f"Episode ended after {episode_step} steps in {episode_time:.1f}s with reward {transition[TransitionKey.REWARD]}"
                )
                episode_step = 0
                episode_idx += 1

                if dataset is not None:
                    if cfg.env.processor.control_mode == "leader" and cfg.mode == "record":
                        # Hold leader while video encode runs (can take several seconds).
                        maybe_hold_leader_teleop(teleop_device)
                    if transition[TransitionKey.INFO].get(TeleopEvents.RERECORD_EPISODE, False):
                        logging.info(f"Re-recording episode {episode_idx}")
                        dataset.clear_episode_buffer()
                        episode_idx -= 1
                    else:
                        logging.info(f"Saving episode {episode_idx}")
                        dataset.save_episode()

                # Reset for new episode
                obs, info = env.reset()
                env_processor.reset()
                action_processor.reset()

                transition = create_transition(observation=obs, info=info)
                transition = env_processor(transition)
                if cfg.env.processor.control_mode == "leader" and cfg.mode == "record":
                    set_intervention = getattr(teleop_device, "_set_intervention", None)
                    if callable(set_intervention):
                        set_intervention(True, release_torque=False)
                    maybe_reset_leader_teleop(teleop_device, env.get_raw_joint_positions())
                    maybe_release_leader_for_teleop(teleop_device, settle_s=2.0)
                if cfg.mode == "record" and episode_idx < cfg.dataset.num_episodes_to_record:
                    episode_step = 0
                    episode_start_time = time.perf_counter()
                    _print_start_recording(episode_idx, cfg.dataset.num_episodes_to_record)

            # Maintain fps timing
            precise_sleep(dt - (time.perf_counter() - step_start_time))

        if dataset is not None and cfg.dataset.push_to_hub:
            logging.info("Pushing dataset to hub")
            dataset.push_to_hub()
    finally:
        if dataset is not None:
            logging.info("Finalizing dataset writers")
            dataset.finalize()


def replay_trajectory(
    env: gym.Env, action_processor: DataProcessorPipeline, cfg: GymManipulatorConfig
) -> None:
    """Replay recorded trajectory on robot environment."""
    assert cfg.dataset.replay_episode is not None, "Replay episode must be provided for replay"

    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=[cfg.dataset.replay_episode],
        download_videos=False,
    )
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == cfg.dataset.replay_episode)
    actions = episode_frames.select_columns(ACTION)

    _, info = env.reset()

    for action_data in actions:
        start_time = time.perf_counter()
        transition = create_transition(
            observation=env.get_raw_joint_positions() if hasattr(env, "get_raw_joint_positions") else {},
            action=action_data[ACTION],
        )
        transition = action_processor(transition)
        env.step(transition[TransitionKey.ACTION])
        precise_sleep(1 / cfg.env.fps - (time.perf_counter() - start_time))


@parser.wrap()
def main(cfg: GymManipulatorConfig) -> None:
    """Main entry point for gym manipulator script."""
    env, teleop_device = make_robot_env(cfg.env)
    env_processor, action_processor = make_processors(env, teleop_device, cfg.env, cfg.device)

    print("Environment observation space:", env.observation_space)
    print("Environment action space:", env.action_space)
    print("Environment processor:", env_processor)
    print("Action processor:", action_processor)

    if cfg.mode == "replay":
        replay_trajectory(env, action_processor, cfg)
        exit()

    control_loop(env, env_processor, action_processor, teleop_device, cfg)


if __name__ == "__main__":
    main()
