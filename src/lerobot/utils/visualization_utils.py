# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import numbers
import os
import socket
from typing import Any

import numpy as np
import rerun as rr

from .constants import OBS_PREFIX, OBS_STR


def _rerun_viewer_reachable(port: int = 9876, timeout_s: float = 0.35) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout_s):
            return True
    except OSError:
        return False


def init_rerun(session_name: str = "lerobot_control_loop", *, port: int | None = None) -> None:
    """Initializes the Rerun SDK for visualizing the control loop.

    If a viewer is already listening on ``port`` (default 9876), reconnect to it
    instead of spawning a new one. That way you can keep the Rerun window open
    across actor restarts, and avoid orphan viewers inheriting camera FDs.
    """
    # Small flush so live camera frames appear promptly (default 8000 can look delayed).
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "1024")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    # Stable recording id → reconnect keeps writing into the same stream (live view).
    recording_id = os.getenv("RERUN_RECORDING_ID", "live")
    rr.init(session_name, recording_id=recording_id)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "25%")
    port = int(os.getenv("RERUN_PORT", str(port if port is not None else 9876)))
    url = os.getenv("RERUN_CONNECT_URL", f"rerun+http://127.0.0.1:{port}/proxy")

    if _rerun_viewer_reachable(port):
        rr.connect_grpc(url)
        print(f"Rerun: reconnecting to existing viewer ({url}), recording_id={recording_id}")
        print("  In the viewer: select recording 'live' and enable Follow (▶) on the timeline")
        return

    rr.spawn(port=port, memory_limit=memory_limit, connect=True)
    print(f"Rerun: spawned new viewer on port {port}, recording_id={recording_id}")
    print("  Paths: observation.cameras.front / observation.cameras.wrist — turn on Follow")


def _is_scalar(x):
    return isinstance(x, (float | numbers.Real | np.integer | np.floating)) or (
        isinstance(x, np.ndarray) and x.ndim == 0
    )


def log_rerun_data(
    observation: dict[str, Any] | None = None,
    action: dict[str, Any] | None = None,
    *,
    step: int | None = None,
) -> None:
    """
    Logs observation and action data to Rerun for real-time visualization.

    This function iterates through the provided observation and action dictionaries and sends their contents
    to the Rerun viewer. It handles different data types appropriately:
    - Scalars values (floats, ints) are logged as `rr.Scalars`.
    - 3D NumPy arrays that resemble images (e.g., with 1, 3, or 4 channels first) are transposed
      from CHW to HWC format and logged as `rr.Image`.
    - 1D NumPy arrays are logged as a series of individual scalars, with each element indexed.
    - Other multi-dimensional arrays are flattened and logged as individual scalars.

    Keys are automatically namespaced with "observation." or "action." if not already present.

    Args:
        observation: An optional dictionary containing observation data to log.
        action: An optional dictionary containing action data to log.
        step: Optional monotonic step index for the timeline (keeps live Follow working).
    """
    if step is not None:
        rr.set_time("step", sequence=int(step))

    if observation:
        for k, v in observation.items():
            if v is None:
                continue
            key = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                arr = v
                # Convert CHW -> HWC when needed
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.ndim == 1:
                    for i, vi in enumerate(arr):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    # Live camera streams must not be static, otherwise the viewer
                    # barely updates during teleop/recording.
                    rr.log(key, rr.Image(arr))

    if action:
        for k, v in action.items():
            if v is None:
                continue
            key = k if str(k).startswith("action.") else f"action.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                if v.ndim == 1:
                    for i, vi in enumerate(v):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    # Fall back to flattening higher-dimensional arrays
                    flat = v.flatten()
                    for i, vi in enumerate(flat):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
