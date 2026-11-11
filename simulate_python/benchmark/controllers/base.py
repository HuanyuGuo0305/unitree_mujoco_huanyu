"""Baseline adapter base class and shared MuJoCo plumbing.

An adapter injects externally generated benchmark commands into an otherwise
unmodified native controller. The only functional difference from the
standalone deployment is where ``base_command`` and ``ee_kp_commands`` come
from: an argument instead of a YAML constant / internal sampler. Everything
downstream of that -- observation assembly, history stacking, ONNX inference,
action scaling, IK state, guidance slew, UAN, torque clipping, actuator
write-out -- is the ported original code.

See benchmark/docs/interface_audit.md section 12.
"""

from __future__ import annotations

import hashlib
import os
from collections import deque

import mujoco
import numpy as np
import onnxruntime as ort

from benchmark.controllers.uan import FrozenUAN, sync_mujoco_arm_limits

LEG_ORDER = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
)
ARM_ORDER = tuple(f"joint{i}" for i in range(1, 7))
WHEEL_ORDER = ("FL_wheel_joint", "FR_wheel_joint", "RL_wheel_joint", "RR_wheel_joint")
GRIPPER_JOINT = "jointGripper"


def file_sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def resolve_repo_path(path: str, repo_root: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(repo_root, path))


def _quat_with_rpy_offset(quat_wxyz, roll: float, pitch: float, yaw: float):
    """Apply small roll/pitch/yaw offsets to a wxyz quaternion, in float64."""
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    for axis, angle in ((0, roll), (1, pitch), (2, yaw)):
        if angle == 0.0:
            continue
        dq = np.zeros(4, dtype=np.float64)
        dq[0] = np.cos(angle / 2.0)
        dq[axis + 1] = np.sin(angle / 2.0)
        w1, x1, y1, z1 = dq
        w2, x2, y2, z2 = q
        q = np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )
    return q / np.linalg.norm(q)


class OnnxPolicy:
    """Single-threaded deterministic ONNX actor (normalizer baked in)."""

    # Thread count changes ONNX Runtime's GEMM reduction order and therefore
    # the last bits of the action, so the benchmark pins it to 1 for
    # bit-reproducible episodes (override with BENCHMARK_ORT_THREADS=0 to use
    # the library default). The parity test pins the standalone scripts the
    # same way, and under that pinning the adapters are bit-exact.
    def __init__(self, path: str, expected_in: int | None = None, expected_out: int | None = None):
        options = ort.SessionOptions()
        n_threads = int(os.environ.get("BENCHMARK_ORT_THREADS", "1"))
        if n_threads > 0:
            options.intra_op_num_threads = n_threads
            options.inter_op_num_threads = n_threads
        self.path = path
        self.sha256 = file_sha256(path)
        self.session = ort.InferenceSession(
            path, options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        in_shape = self.session.get_inputs()[0].shape
        out_shape = self.session.get_outputs()[0].shape
        if expected_in is not None and int(in_shape[-1]) != expected_in:
            raise ValueError(f"{path}: expected input dim {expected_in}, got {in_shape}")
        if expected_out is not None and int(out_shape[-1]) != expected_out:
            raise ValueError(f"{path}: expected output dim {expected_out}, got {out_shape}")
        self.n_calls = 0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.ascontiguousarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        out = self.session.run([self.output_name], {self.input_name: obs})[0]
        self.n_calls += 1
        return np.asarray(out, dtype=np.float32).reshape(-1)


class TermMajorHistory:
    """Isaac Lab observation group: each term keeps its own oldest->newest deque."""

    def __init__(self, schema, length: int):
        self.schema = tuple(schema)
        self.length = int(length)
        self.hist = {name: deque(maxlen=self.length) for name, _ in self.schema}

    def seed(self, values: dict):
        self.hist = {name: deque(maxlen=self.length) for name, _ in self.schema}
        for name, dim in self.schema:
            v = np.asarray(values[name], dtype=np.float32).reshape(-1)
            if v.shape != (dim,):
                raise ValueError(f"Term {name}: expected ({dim},), got {v.shape}")
            for _ in range(self.length):
                self.hist[name].append(v.copy())

    def append(self, values: dict):
        for name, dim in self.schema:
            v = np.asarray(values[name], dtype=np.float32).reshape(-1)
            if v.shape != (dim,):
                raise ValueError(f"Term {name}: expected ({dim},), got {v.shape}")
            self.hist[name].append(v.copy())

    def stack(self) -> np.ndarray:
        for name, _ in self.schema:
            if len(self.hist[name]) != self.length:
                raise RuntimeError(f"History {name} not initialised")
        return np.concatenate(
            [
                np.asarray(self.hist[name], dtype=np.float32).reshape(-1)
                for name, _ in self.schema
            ]
        ).astype(np.float32)


class BaselineAdapter:
    """Common lifecycle for every baseline adapter.

    Subclasses must set ``baseline_id``, ``has_wheels``, ``blend_before_step``
    and implement ``_build_terms``, ``_policy_inference`` and ``_apply_action``.
    """

    baseline_id: str = "?"
    has_wheels: bool = True
    # C1/C2 compute the startup blend from sim_time BEFORE the physics step;
    # C3-C6 compute it inside the policy tick from the incremented sim_time.
    blend_before_step: bool = False
    # Dtype of the control path. C1/C2 cast the joint states to float32 and run
    # the whole PD/target path in float32 (b2wz1_wbc_plb_uan.py:931-955);
    # C3-C6 keep float64. Matching this exactly is required for bit-level
    # parity with the original scripts -- see docs/interface_audit.md 12.
    state_dtype = np.float64
    # tau_pd dtype inside the frozen UAN (see FrozenUAN.pd_dtype).
    uan_pd_dtype = np.float32
    # Gripper gain dtype. C1 indexes a float32 kps array, so its gripper
    # product is rounded to float32 (b2wz1_wbc_plb_uan.py:950); C2-C6 wrap the
    # gain in float(), making the whole gripper law float64
    # (b2wz1_wbc_plb_legged_uan.py:628). One float32 ULP of gripper torque is
    # enough to break bit-exactness, so the two cases are kept distinct.
    gripper_gain_dtype = np.float64

    def __init__(self, cfg: dict, cfg_path: str, repo_root: str, verbose: bool = False):
        self.cfg = cfg
        self.cfg_path = cfg_path
        self.repo_root = repo_root
        self.verbose = verbose

        self.simulation_dt = float(cfg["simulation_dt"])
        self.control_decimation = int(cfg["control_decimation"])
        self.control_dt = self.simulation_dt * self.control_decimation
        self.history_length = int(cfg["history_length"])
        if abs(self.simulation_dt - 0.004) > 1e-9 or self.control_decimation != 5:
            raise ValueError(
                f"{self.baseline_id}: expected simulation_dt=0.004 and decimation=5, "
                f"got {self.simulation_dt} / {self.control_decimation}"
            )
        if self.history_length != 5:
            raise ValueError(f"{self.baseline_id}: expected history_length=5")

        self.startup_hold_s = float(cfg.get("startup_hold_s", 1.0))
        self.startup_blend_s = float(cfg.get("startup_blend_s", 2.0))

        self.xml_path = resolve_repo_path(cfg["xml_path"], repo_root)
        self.uan_model_path = resolve_repo_path(cfg["uan_model_path"], repo_root)
        self.ee_body_name = str(cfg.get("ee_body", "gripperStator"))
        self.ee_ground_z = float(cfg.get("ee_ground_z", cfg.get("ground_z", 0.0)))
        self.ee_kp_dx = float(cfg.get("ee_kp_dx", 0.30))
        self.ee_kp_dz = float(cfg.get("ee_kp_dz", 0.30))

        # The initial state is written at the control dtype, because C1/C2
        # store root_pos/root_quat/default_joint_pos as float32
        # (b2wz1_wbc_plb_uan.py:289-293) while C3-C6 use float64. The float32
        # rounding of e.g. 0.6017 and 1.48 is a ~1e-8 difference that, left
        # unmatched, diverges from the original run within a few seconds.
        self.root_pos = np.asarray(cfg["root_pos"], dtype=self.state_dtype).reshape(3)
        self.root_quat = np.asarray(
            cfg["root_quat_wxyz"], dtype=self.state_dtype
        ).reshape(4)
        # float64 copies, used only by apply_initial_state_noise so that the
        # randomised initial state is identical for every baseline.
        self.root_pos_f64 = np.asarray(cfg["root_pos"], dtype=np.float64).reshape(3)
        self.root_quat_f64 = np.asarray(
            cfg["root_quat_wxyz"], dtype=np.float64
        ).reshape(4)

        self.leg_joint_names = list(LEG_ORDER)
        self.arm_joint_names = list(ARM_ORDER)
        self.wheel_joint_names = list(WHEEL_ORDER) if self.has_wheels else []
        self.gripper_joint_name = GRIPPER_JOINT

        self.model = None
        self.data = None
        self.uan = None
        self._blend = 0.0
        self.policy_ticks = 0
        self.physics_steps = 0
        self.last_base_command = np.zeros(3, dtype=np.float32)
        self.last_ee_command = np.zeros(9, dtype=np.float32)
        self.asset_hashes: dict = {}

    # ---------------------------------------------------------------- setup
    def build_model(self):
        model = mujoco.MjModel.from_xml_path(self.xml_path)
        model.opt.timestep = self.simulation_dt
        return model

    def attach(self, model, data):
        self.model = model
        self.data = data
        m = model

        self._config_joint_names = self.cfg.get("config_joint_names")
        if self._config_joint_names is None:
            # C1's YAML arrays are indexed by the MuJoCo 23-joint qpos order.
            self._config_joint_names = [
                "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_wheel_joint",
                "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_wheel_joint",
                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RL_wheel_joint",
                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RR_wheel_joint",
                "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "jointGripper",
            ]
        cfg_index = {name: i for i, name in enumerate(self._config_joint_names)}

        dt = self.state_dtype

        def cfg_array(key, names):
            values = self.cfg[key]
            return np.asarray([values[cfg_index[n]] for n in names], dtype=dt)

        all_names = (
            self.leg_joint_names
            + self.arm_joint_names
            + self.wheel_joint_names
            + [self.gripper_joint_name]
        )
        self.qpos_addr, self.qvel_addr, self.actuator_for_joint = {}, {}, {}
        for name in all_names:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint not found in {self.xml_path}: {name}")
            self.qpos_addr[name] = int(m.jnt_qposadr[jid])
            self.qvel_addr[name] = int(m.jnt_dofadr[jid])
        for act_id in range(m.nu):
            trn = int(m.actuator_trnid[act_id, 0])
            jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, trn)
            if jname in all_names:
                self.actuator_for_joint[jname] = act_id
        missing = [n for n in all_names if n not in self.actuator_for_joint]
        if missing:
            raise ValueError(f"No actuator for joints {missing} in {self.xml_path}")

        self.leg_qidx = [self.qpos_addr[n] for n in self.leg_joint_names]
        self.leg_vidx = [self.qvel_addr[n] for n in self.leg_joint_names]
        self.arm_qidx = [self.qpos_addr[n] for n in self.arm_joint_names]
        self.arm_vidx = [self.qvel_addr[n] for n in self.arm_joint_names]
        self.wheel_qidx = [self.qpos_addr[n] for n in self.wheel_joint_names]
        self.wheel_vidx = [self.qvel_addr[n] for n in self.wheel_joint_names]
        self.gripper_qidx = self.qpos_addr[self.gripper_joint_name]
        self.gripper_vidx = self.qvel_addr[self.gripper_joint_name]

        self.leg_act = [self.actuator_for_joint[n] for n in self.leg_joint_names]
        self.arm_act = [self.actuator_for_joint[n] for n in self.arm_joint_names]
        self.wheel_act = [self.actuator_for_joint[n] for n in self.wheel_joint_names]
        self.gripper_act = self.actuator_for_joint[self.gripper_joint_name]

        self.default_by_name = {
            n: dt(self.cfg["default_joint_pos"][cfg_index[n]]) for n in all_names
        }
        self.default_f64_by_name = {
            n: float(self.cfg["default_joint_pos"][cfg_index[n]]) for n in all_names
        }
        self.leg_default = cfg_array("default_joint_pos", self.leg_joint_names)
        self.arm_default = cfg_array("default_joint_pos", self.arm_joint_names)
        # The gripper error is formed from PYTHON floats in every script
        # (float(default) - float(d.qpos[...])), so the subtraction happens in
        # float64 and only the product with the float32 gain is rounded. Doing
        # the subtraction in float32 instead costs ~1 float32 ULP and breaks
        # bit-exactness against the originals.
        self.gripper_default = float(self.default_by_name[self.gripper_joint_name])

        self.leg_kp = cfg_array("kps", self.leg_joint_names)
        self.leg_kd = cfg_array("kds", self.leg_joint_names)
        gdt = self.gripper_gain_dtype
        self.gripper_kp = gdt(self.cfg["kps"][cfg_index[self.gripper_joint_name]])
        self.gripper_kd = gdt(self.cfg["kds"][cfg_index[self.gripper_joint_name]])

        self.leg_torque_limits = np.asarray(
            self.cfg["leg_torque_limits"], dtype=dt
        ).reshape(12)
        self.gripper_torque_limit = float(self.cfg.get("gripper_torque_limit", 30.0))
        self.leg_action_scale = float(self.cfg["leg_action_scale"])
        if self.has_wheels:
            self.wheel_action_scale = float(self.cfg["wheel_action_scale"])
            self.wheel_velocity_limits = np.asarray(
                self.cfg["wheel_velocity_limits"], dtype=dt
            ).reshape(4)

        self.ee_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self.ee_body_name)
        if self.ee_bid < 0:
            raise ValueError(f"Body not found: {self.ee_body_name}")
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro") < 0:
            raise ValueError("MuJoCo XML is missing the required 'imu_gyro' sensor")
        self.gyro_adr = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")])

        self.uan = FrozenUAN(
            self.uan_model_path, self.cfg, pd_dtype=self.uan_pd_dtype
        )
        sync_mujoco_arm_limits(
            m,
            self.arm_joint_names,
            self.actuator_for_joint,
            self.uan.final_limits,
            enabled=bool(self.cfg.get("sync_mujoco_arm_limits_to_uan", True)),
            verbose=self.verbose,
        )

        self.asset_hashes = {
            "xml": file_sha256(self.xml_path),
            "uan_model": file_sha256(self.uan_model_path),
        }
        self._load_policies()
        self._setup()

    def _load_policies(self):
        raise NotImplementedError

    def _setup(self):
        pass

    # ---------------------------------------------------------------- reset
    def set_initial_state(self):
        d, m = self.data, self.model
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.qacc[:] = 0.0
        d.ctrl[:] = 0.0
        d.qpos[0:3] = self.root_pos
        d.qpos[3:7] = self.root_quat
        for name, value in self.default_by_name.items():
            d.qpos[self.qpos_addr[name]] = value
        mujoco.mj_forward(m, d)

    def apply_initial_state_noise(self, seed: int, cfg: dict):
        """Seeded perturbation of the initial state, applied AFTER
        ``set_initial_state``.

        A pure function of the seed, so every baseline at a given seed starts
        from exactly the same state. Two details make that exact:

        * the perturbed state is rebuilt from the YAML values in **float64**
          for every baseline, bypassing the per-baseline control dtype. Without
          this, C1/C2 (whose scripts hold the initial state in float32) would
          start ~1e-8 away from C3-C6 and the "same" episode would not be the
          same episode. ``set_initial_state`` itself stays dtype-faithful,
          because the parity tests must reproduce the originals exactly and
          they never call this method.
        * wheel joints are skipped, so the wheeled and fixed-wheel models get
          identical perturbations from the same draw.
        """
        import hashlib

        import mujoco

        if not cfg or not cfg.get("enabled", False):
            return {"enabled": False, "note": "nominal initial state, per-baseline dtype"}

        digest = hashlib.sha256(f"{int(seed)}:initial_state".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))

        d = self.data
        q_noise = float(cfg.get("joint_pos_noise", 0.0))
        v_noise = float(cfg.get("joint_vel_noise", 0.0))
        z_noise = float(cfg.get("base_z_noise", 0.0))
        rpy_noise = np.radians(float(cfg.get("base_rpy_noise_deg", 0.0)))
        yaw_noise = np.radians(float(cfg.get("base_yaw_noise_deg", 0.0)))

        # Fixed joint list, so the draw never depends on the model's joint count.
        names = self.leg_joint_names + self.arm_joint_names + [self.gripper_joint_name]
        dq = rng.uniform(-q_noise, q_noise, len(names))
        dv = rng.uniform(-v_noise, v_noise, len(names))
        dz = rng.uniform(0.0, z_noise)
        droll, dpitch = rng.uniform(-rpy_noise, rpy_noise, 2)
        dyaw = rng.uniform(-yaw_noise, yaw_noise)

        for name, dqi, dvi in zip(names, dq, dv):
            d.qpos[self.qpos_addr[name]] = self.default_f64_by_name[name] + float(dqi)
            d.qvel[self.qvel_addr[name]] = float(dvi)
        for name in self.wheel_joint_names:
            d.qpos[self.qpos_addr[name]] = self.default_f64_by_name[name]
            d.qvel[self.qvel_addr[name]] = 0.0

        d.qpos[0:3] = self.root_pos_f64
        d.qpos[2] = self.root_pos_f64[2] + float(dz)
        d.qpos[3:7] = _quat_with_rpy_offset(self.root_quat_f64, droll, dpitch, dyaw)
        d.qvel[0:6] = 0.0
        mujoco.mj_forward(self.model, d)
        return {
            "enabled": True,
            "dtype": "float64 (identical for every baseline)",
            "joint_order": names,
            "joint_pos_delta": dq.tolist(),
            "joint_vel_delta": dv.tolist(),
            "base_z_delta": float(dz),
            "base_roll_delta_rad": float(droll),
            "base_pitch_delta_rad": float(dpitch),
            "base_yaw_delta_rad": float(dyaw),
        }

    def reset(self, base_command: np.ndarray, ee_command: np.ndarray):
        """Reset controller state. ``*_command`` seed the observation histories."""
        self.uan.reset()
        self._blend = 0.0
        self.policy_ticks = 0
        self.physics_steps = 0
        self.last_base_command = np.asarray(base_command, dtype=np.float32).reshape(3).copy()
        self.last_ee_command = np.asarray(ee_command, dtype=np.float32).reshape(9).copy()
        self.leg_target = self.leg_default.copy()
        self.arm_target = self.arm_default.copy()
        self.gripper_target = float(self.gripper_default)
        if self.has_wheels:
            self.wheel_cmd = np.zeros(4, dtype=self.state_dtype)
        self._reset_controller()

    def _reset_controller(self):
        raise NotImplementedError

    # -------------------------------------------------------------- stepping
    def blend_at(self, sim_time: float) -> float:
        if sim_time < self.startup_hold_s:
            return 0.0
        if self.startup_blend_s > 0.0:
            return float(
                np.clip((sim_time - self.startup_hold_s) / self.startup_blend_s, 0.0, 1.0)
            )
        return 1.0

    def pre_physics(self, sim_time: float):
        """Compute torques from the held targets and write ``d.ctrl``."""
        d = self.data
        if self.blend_before_step:
            self._blend = self.blend_at(sim_time)

        dt = self.state_dtype
        leg_pos = d.qpos[self.leg_qidx].astype(dt)
        leg_vel = d.qvel[self.leg_vidx].astype(dt)
        arm_pos = d.qpos[self.arm_qidx].astype(dt)
        arm_vel = d.qvel[self.arm_vidx].astype(dt)
        # The joint state is rounded to the control dtype first (C1/C2 read it
        # through a float32 view) and only then widened to a python float.
        grip_pos = float(dt(d.qpos[self.gripper_qidx]))
        grip_vel = float(dt(d.qvel[self.gripper_vidx]))

        leg_tau = np.clip(
            self.leg_kp * (self.leg_target - leg_pos) - self.leg_kd * leg_vel,
            -self.leg_torque_limits,
            self.leg_torque_limits,
        ).astype(dt)
        arm_tau = self.uan.torque(self.arm_target, arm_pos, arm_vel)
        grip_tau = float(
            np.clip(
                self.gripper_kp * (self.gripper_target - grip_pos) - self.gripper_kd * grip_vel,
                -self.gripper_torque_limit,
                self.gripper_torque_limit,
            )
        )

        d.ctrl[:] = 0.0
        d.ctrl[self.leg_act] = leg_tau
        d.ctrl[self.arm_act] = arm_tau
        d.ctrl[self.gripper_act] = grip_tau
        if self.has_wheels:
            d.ctrl[self.wheel_act] = np.clip(
                self.wheel_cmd, -self.wheel_velocity_limits, self.wheel_velocity_limits
            ).astype(dt)
        self.physics_steps += 1

    def policy_tick(self, base_command: np.ndarray, ee_command: np.ndarray, sim_time: float):
        """One 50 Hz control update with externally supplied commands."""
        base_command = np.asarray(base_command, dtype=np.float32).reshape(3)
        ee_command = np.asarray(ee_command, dtype=np.float32).reshape(9)
        self.last_base_command = base_command.copy()
        self.last_ee_command = ee_command.copy()
        if not self.blend_before_step:
            self._blend = self.blend_at(sim_time)
        self._policy_update(base_command, ee_command)
        self.policy_ticks += 1

    def _policy_update(self, base_command: np.ndarray, ee_command: np.ndarray):
        raise NotImplementedError

    # -------------------------------------------------------- shared obs bits
    def _base_ang_vel(self) -> np.ndarray:
        return self.data.sensordata[self.gyro_adr : self.gyro_adr + 3].astype(np.float32).copy()

    def _projected_gravity(self) -> np.ndarray:
        from utilities.math import quat_rotate_inverse_numpy, quat_unique_wxyz

        q = quat_unique_wxyz(self.data.qpos[3:7].copy().astype(np.float32))
        return quat_rotate_inverse_numpy(
            q, np.array([0.0, 0.0, -1.0], dtype=np.float32)
        ).astype(np.float32)

    def last_policy_action(self) -> np.ndarray:
        """Raw policy action as stored in the observation (for parity tests)."""
        return self.last_action

    def diagnostics(self) -> dict:
        return {
            "uan_action_saturation": float(
                np.mean(np.abs(self.uan.last_action) > self.uan.saturation_threshold)
            ),
            "uan_delta_tau_abs_mean": float(np.mean(np.abs(self.uan.last_delta_tau))),
            "blend": float(self._blend),
        }
