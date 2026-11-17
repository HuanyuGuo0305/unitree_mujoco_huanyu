"""High-level methods for E4: No Residual, EE Position Only, EE Orientation Only, Full.

Each learned method mirrors the Isaac Lab environment it was trained in:

  method          ONNX                 obs/frame  history  action   decoder
  full            high_policy_2.onnx   64         3        6        b2wz1_hl_transport_UAN
  pos_only        high_policy_pos.onnx 61         3        3 (xyz)  b2wz1_hl_transport_pos_only_UAN
  ori_only        high_policy_orn.onnx 61         3        3 (ypr)  b2wz1_hl_transport_ori_only_UAN
  no_residual     --                   --         --       --       zero residual (same clipping/settle hold)

The 3-D policies are their OWN independently trained networks (never a masked
6-D policy). Their previous-action observation is the 3-D executed residual of
their own channels. ``diag6`` is a zero-filled 6-D diagnostic copy for logging
only and is never fed back to any policy.
"""
from __future__ import annotations

import hashlib

import numpy as np
import onnxruntime as ort

from .sim import FeatureMajorHistory, resolve_repo_path

METHODS = ("no_residual", "pos_only", "ori_only", "full")
LABELS = {
    "no_residual": "No Residual",
    "pos_only": "EE Position Only",
    "ori_only": "EE Orientation Only",
    "full": "Full Policy (Ours)",
}
ACTION_DIM = {"no_residual": 0, "pos_only": 3, "ori_only": 3, "full": 6}


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hl_feature_dims(prev_action_dim: int):
    # ang vel, gravity, leg q, arm q, leg qd, arm qd, wheel qd, base ref, EE pos ref,
    # EE ori ref, previous executed action, puck center in disk frame.
    return [3, 3, 12, 6, 12, 6, 4, 3, 3, 3, prev_action_dim, 3]


def decode_residual(method: str, action, sim):
    """Decode a normalized residual against the CURRENT references.

    Returns (raw, effective, kp0_cmd, rot_cmd, executed, diag6). Arithmetic is
    identical to ``_decode_hl_action_to_ll_commands`` of each Isaac Lab variant
    (and to the deploy script for ``full``).
    """
    adim = ACTION_DIM[method]
    if adim == 0:
        raw = np.zeros(0, dtype=np.float32)
    else:
        action = np.asarray(action, dtype=np.float32).reshape(adim,)
        if not np.isfinite(action).all():
            raise FloatingPointError(f"Non-finite high-level action: {action}")
        raw = np.clip(action, -1.0, 1.0).astype(np.float32)
    eff = raw.copy()
    if sim.settling:
        eff[:] = 0.0

    ref = sim.ee_kp0_ref
    rot_ref = sim.rot_ref
    if method == "full":
        pos_raw = ref + eff[0:3] * sim.ee_pos_scale
        rot_raw = rot_ref + eff[3:6] * sim.ee_rot_scale
    elif method == "pos_only":
        pos_raw = ref + eff[0:3] * sim.ee_pos_scale
        rot_raw = rot_ref                                   # orientation residual == 0
    elif method == "ori_only":
        pos_raw = ref                                       # position residual == 0
        rot_raw = rot_ref + eff[0:3] * sim.ee_rot_scale
    elif method == "no_residual":
        pos_raw = ref
        rot_raw = rot_ref
    else:
        raise ValueError(method)
    kp0_cmd = np.clip(pos_raw, sim.ee_pos_cmd_min, sim.ee_pos_cmd_max).astype(np.float32)
    if sim.settling:
        kp0_cmd = ref.copy()                                # measured-kp0 hold during settle
    rot_cmd = np.clip(rot_raw, sim.ee_rot_cmd_min, sim.ee_rot_cmd_max).astype(np.float32)

    exec_pos = np.clip((kp0_cmd - ref) / sim.ee_pos_scale, -1.0, 1.0).astype(np.float32)
    exec_rot = np.clip((rot_cmd - rot_ref) / sim.ee_rot_scale, -1.0, 1.0).astype(np.float32)
    zeros3 = np.zeros(3, dtype=np.float32)
    if method == "full":
        executed = np.clip(np.concatenate([(kp0_cmd - ref) / sim.ee_pos_scale,
                                           (rot_cmd - rot_ref) / sim.ee_rot_scale]), -1.0, 1.0).astype(np.float32)
        diag6 = executed.copy()
    elif method == "pos_only":
        executed = exec_pos
        diag6 = np.concatenate([exec_pos, zeros3])
    elif method == "ori_only":
        executed = exec_rot
        diag6 = np.concatenate([zeros3, exec_rot])
    else:
        executed = np.zeros(0, dtype=np.float32)
        diag6 = np.zeros(6, dtype=np.float32)
    return raw, eff, kp0_cmd, rot_cmd, executed, diag6


class HLMethod:
    """One evaluation method bound to a TransportSim."""

    def __init__(self, method: str, policy_path: str | None, expected_sha256: str | None = None,
                 hl_history_length: int = 3, ort_threads: int = 1):
        if method not in METHODS:
            raise ValueError(method)
        self.method = method
        self.action_dim = ACTION_DIM[method]
        self.policy_path = None
        self.policy_sha256 = None
        self.sess = None
        if self.action_dim > 0:
            if policy_path is None:
                raise ValueError(f"{method} requires a policy file")
            self.policy_path = resolve_repo_path(policy_path)
            self.policy_sha256 = file_sha256(self.policy_path)
            if expected_sha256 and self.policy_sha256 != expected_sha256:
                raise ValueError(f"{method}: policy hash {self.policy_sha256} != expected {expected_sha256}")
            so = ort.SessionOptions()
            if ort_threads > 0:
                so.intra_op_num_threads = ort_threads
                so.inter_op_num_threads = ort_threads
            self.sess = ort.InferenceSession(self.policy_path, sess_options=so, providers=["CPUExecutionProvider"])
            self.in_name = self.sess.get_inputs()[0].name
            self.out_name = self.sess.get_outputs()[0].name
            self.dims = hl_feature_dims(self.action_dim)
            self.frame_dim = sum(self.dims)
            self.obs_dim = self.frame_dim * hl_history_length
            in_dim = self.sess.get_inputs()[0].shape[-1]
            out_dim = self.sess.get_outputs()[0].shape[-1]
            if in_dim != self.obs_dim or out_dim != self.action_dim:
                raise ValueError(f"{method}: ONNX {in_dim}->{out_dim} but expected {self.obs_dim}->{self.action_dim}")
            self.history = FeatureMajorHistory(self.dims, hl_history_length)
        self.executed = np.zeros(self.action_dim, dtype=np.float32)

    def reset(self, sim):
        self.executed = np.zeros(self.action_dim, dtype=np.float32)
        self._first = True
        if self.sess is not None:
            frame = sim.hl_obs_frame(self.executed)
            assert frame.shape == (self.frame_dim,)
            self.history.fill(frame)

    def policy_action(self, sim):
        if self.sess is None:
            return None
        if not self._first:
            # obs_(t+1): current refs + a_t^exec (executed in the step just completed).
            self.history.append(sim.hl_obs_frame(self.executed))
        self._first = False
        obs = self.history.flat()
        return self.sess.run([self.out_name], {self.in_name: obs[None, :]})[0][0].astype(np.float32)

    def step(self, sim):
        """Infer, decode and install the EE command for the next 10-Hz transition."""
        action = self.policy_action(sim)
        raw, eff, kp0_cmd, rot_cmd, executed, diag6 = decode_residual(self.method, action, sim)
        self.executed = executed
        sim.set_ee_command(kp0_cmd, rot_cmd)
        return {"raw": raw, "effective": eff, "executed": executed, "diag6": diag6,
                "kp0_cmd": kp0_cmd, "rot_cmd": rot_cmd}


def build_methods(exp_cfg: dict, hl_history_length: int = 3):
    out = {}
    for name in METHODS:
        spec = exp_cfg["methods"][name]
        out[name] = HLMethod(name, spec.get("policy"), spec.get("sha256"), hl_history_length)
    return out
