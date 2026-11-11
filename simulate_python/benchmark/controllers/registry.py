"""Baseline registry: ID -> (adapter class, native YAML config)."""

from __future__ import annotations

import os

import yaml

from benchmark.controllers.c1_wbc import C1Adapter, C2Adapter
from benchmark.controllers.c3_vbc import C3Adapter, C4Adapter
from benchmark.controllers.c5_roboduet import C5Adapter, C6Adapter

SIM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(SIM_ROOT, ".."))

BASELINES = {
    "C1": {
        "adapter": C1Adapter,
        "config": "configs/b2wz1_wbc_plb_uan.yaml",
        "framework": "Unified whole-body PPO (ours)",
        "actuation": "wheeled-legged",
        "label": "C1 Unified RL",
        "deploy_script": "deploy_mujoco/b2wz1_wbc_plb_uan.py",
    },
    "C2": {
        "adapter": C2Adapter,
        "config": "configs/b2wz1_wbc_plb_legged_uan.yaml",
        "framework": "Unified whole-body PPO (ours)",
        "actuation": "fixed-wheel",
        "label": "C2 Unified RL (fixed wheel)",
        "deploy_script": "deploy_mujoco/b2wz1_wbc_plb_legged_uan.py",
    },
    "C3": {
        "adapter": C3Adapter,
        "config": "configs/b2wz1_vbc_plb_uan.yaml",
        "framework": "VBC-inspired locomotion PPO + stateful arm IK",
        "actuation": "wheeled-legged",
        "label": "C3 RL + IK",
        "deploy_script": "deploy_mujoco/b2wz1_vbc_plb_uan.py",
    },
    "C4": {
        "adapter": C4Adapter,
        "config": "configs/b2wz1_vbc_plb_legged_uan.yaml",
        "framework": "VBC-inspired locomotion PPO + stateful arm IK",
        "actuation": "fixed-wheel",
        "label": "C4 RL + IK (fixed wheel)",
        "deploy_script": "deploy_mujoco/b2wz1_vbc_plb_legged_uan.py",
    },
    "C5": {
        "adapter": C5Adapter,
        "config": "configs/b2wz1_roboduet_plb_uan.yaml",
        "framework": "RoboDuet-inspired cooperative Dual-PPO",
        "actuation": "wheeled-legged",
        "label": "C5 Dual-PPO",
        "deploy_script": "deploy_mujoco/b2wz1_roboduet_plb_uan.py",
    },
    "C6": {
        "adapter": C6Adapter,
        "config": "configs/b2wz1_roboduet_plb_legged_uan.yaml",
        "framework": "RoboDuet-inspired cooperative Dual-PPO",
        "actuation": "fixed-wheel",
        "label": "C6 Dual-PPO (fixed wheel)",
        "deploy_script": "deploy_mujoco/b2wz1_roboduet_plb_legged_uan.py",
    },
}


def baseline_label(baseline_id: str) -> str:
    return BASELINES[baseline_id]["label"]


def config_path(baseline_id: str) -> str:
    return os.path.join(SIM_ROOT, BASELINES[baseline_id]["config"])


def load_baseline_config(baseline_id: str) -> tuple[dict, str]:
    path = config_path(baseline_id)
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise TypeError(f"{path}: expected a YAML mapping")
    return cfg, path


def build_adapter(baseline_id: str, verbose: bool = False):
    """Construct an adapter from the baseline's *own* native YAML config."""
    if baseline_id not in BASELINES:
        raise KeyError(f"Unknown baseline '{baseline_id}'. Known: {sorted(BASELINES)}")
    cfg, path = load_baseline_config(baseline_id)
    adapter_cls = BASELINES[baseline_id]["adapter"]
    return adapter_cls(cfg, path, REPO_ROOT, verbose=verbose)
