"""
Sim2Sim deployment for Unitree B2 robot in MuJoCo.
Matches Isaac Lab training configuration.
"""
import time
import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml
from pathlib import Path


def quaternion_to_gravity(quat):
    """Convert quaternion [w,x,y,z] to projected gravity in body frame."""
    qw, qx, qy, qz = quat
    gx = 2 * (qx * qz - qw * qy)
    gy = 2 * (qy * qz + qw * qx)
    gz = qw * qw - qx * qx - qy * qy + qz * qz

    return np.array([gx, gy, -gz])


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD controller: tau = kp*(target_q - q) + kd*(target_dq - dq)"""
    return (target_q - q) * kp + (target_dq - dq) * kd


class B2Controller:
    """Unitree B2 controller for MuJoCo simulation."""
    
    def __init__(self, config_path):
        # Load config
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        base_dir = Path(__file__).parent.parent
        self.policy_path = base_dir / cfg['policy_path']
        self.xml_path = base_dir / cfg['xml_path']
        
        # Simulation parameters
        self.sim_dt = cfg['simulation_dt']
        self.control_decimation = cfg['control_decimation']
        self.sim_duration = cfg['simulation_duration']
        
        # Control gains
        self.kps = np.array(cfg['kps'], dtype=np.float32)
        self.kds = np.array(cfg['kds'], dtype=np.float32)
        self.default_angles = np.array(cfg['default_angles'], dtype=np.float32)
        
        # Observation/action scales
        self.ang_vel_scale = cfg.get('ang_vel_scale', 1.0)
        self.dof_pos_scale = cfg.get('dof_pos_scale', 1.0)
        self.dof_vel_scale = cfg.get('dof_vel_scale', 1.0)
        self.action_scale = cfg.get('action_scale', 0.25)
        self.cmd_scale = np.array(cfg.get('cmd_scale', [1.0, 1.0, 1.0]), dtype=np.float32)
        
        # Dimensions
        self.num_actions = cfg['num_actions']
        self.num_obs = cfg['num_obs']
        
        # Command and state
        self.cmd = np.array(cfg['cmd_init'], dtype=np.float32)
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.counter = 0
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.sim_dt
        
        # Load policy
        self.policy = torch.jit.load(str(self.policy_path))
        self.policy.eval()
        
        # Setup joint mapping (policy order <-> MuJoCo order)
        self._setup_joint_mapping()
        
        # Load observation normalization
        self.use_normalization = cfg.get('use_obs_normalization', False)
        if self.use_normalization:
            norm_dir = self.policy_path.parent
            mean_path = norm_dir / 'obs_mean.npy'
            std_path = norm_dir / 'obs_std.npy'
            
            if mean_path.exists() and std_path.exists():
                self.obs_mean = np.load(mean_path).astype(np.float32)
                self.obs_std = np.load(std_path).astype(np.float32)
                print(f"Loaded normalization: mean={self.obs_mean.shape}, std={self.obs_std.shape}")
            else:
                print(f"Normalization files not found, disabling normalization")
                self.use_normalization = False
        
        print(f"\n{'='*70}")
        print(f"B2 Controller Initialized")
        print(f"{'='*70}")
        print(f"Control freq: {1.0/(self.sim_dt*self.control_decimation):.1f} Hz")
        print(f"Use normalization: {self.use_normalization}")
        print(f"{'='*70}\n")
    
    def _setup_joint_mapping(self):
        """Map between policy joint order and MuJoCo joint order."""
        policy_names = [
            'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
            'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
            'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'
        ]
        
        mujoco_names = []
        for i in range(self.model.nu):
            joint_id = self.model.actuator_trnid[i, 0]
            mujoco_names.append(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
        
        # Create mapping: policy_idx -> mujoco_idx
        self.policy_to_mujoco = np.array([mujoco_names.index(name) for name in policy_names])
        
        # Convert default angles to MuJoCo order
        self.default_angles_mj = np.zeros(self.num_actions, dtype=np.float32)
        for pol_idx in range(self.num_actions):
            mj_idx = self.policy_to_mujoco[pol_idx]
            self.default_angles_mj[mj_idx] = self.default_angles[pol_idx]
    
    def reset(self):
        """Reset robot to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.5  # Initial height
        self.data.qpos[7:] = self.default_angles_mj
        mujoco.mj_forward(self.model, self.data)
        
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.counter = 0
        
        print(f"Robot reset: height={self.data.qpos[2]:.3f}m\n")
    
    def get_observation(self):
        """
        Construct observation vector (45 dims):
        [0:3]   base_ang_vel
        [3:6]   projected_gravity
        [6:9]   velocity_commands
        [9:21]  joint_pos (relative to default)
        [21:33] joint_vel
        [33:45] last_action
        """
        # Base state
        quat = self.data.qpos[3:7]
        ang_vel = self.data.qvel[3:6]
        gravity = quaternion_to_gravity(quat)
        
        # Joint states (MuJoCo order)
        joint_pos_mj = self.data.qpos[7:]
        joint_vel_mj = self.data.qvel[6:]
        
        # Convert to policy order
        joint_pos = joint_pos_mj[self.policy_to_mujoco]
        joint_vel = joint_vel_mj[self.policy_to_mujoco]
        
        # Scale
        joint_pos_rel = (joint_pos - self.default_angles) * self.dof_pos_scale
        joint_vel_scaled = joint_vel * self.dof_vel_scale
        ang_vel_scaled = ang_vel * self.ang_vel_scale
        
        # Assemble observation
        self.obs[0:3] = ang_vel_scaled
        self.obs[3:6] = gravity
        self.obs[6:9] = self.cmd * self.cmd_scale
        self.obs[9:21] = joint_pos_rel
        self.obs[21:33] = joint_vel_scaled
        self.obs[33:45] = self.action
        
        # Normalize
        if self.use_normalization:
            self.obs = (self.obs - self.obs_mean) / (self.obs_std + 1e-8)
        
        return self.obs
    
    def compute_control(self):
        """Compute target joint positions from policy."""
        obs = self.get_observation()
        
        # Policy inference
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
            action_tensor = self.policy(obs_tensor)
            self.action = action_tensor.squeeze(0).numpy().astype(np.float32)
        
        # # Clip actions
        # self.action = np.clip(self.action, -1.0, 1.0)
        
        # Convert to target positions (policy order)
        target_pos = self.action * self.action_scale + self.default_angles
        
        # Convert to MuJoCo order
        target_pos_mj = np.zeros(self.num_actions, dtype=np.float32)
        for pol_idx in range(self.num_actions):
            mj_idx = self.policy_to_mujoco[pol_idx]
            target_pos_mj[mj_idx] = target_pos[pol_idx]
        
        return target_pos_mj
    
    def run(self):
        """Run simulation with viewer."""
        self.reset()
        print(f"Starting simulation ({self.sim_duration:.0f}s)... Press ESC to quit\n")
        
        target_pos = self.default_angles_mj.copy()
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start_time = time.time()
            last_print = start_time
            
            while viewer.is_running():
                step_start = time.time()
                elapsed = time.time() - start_time
                
                if elapsed >= self.sim_duration:
                    break
                
                # PD control
                tau = pd_control(target_pos, self.data.qpos[7:], self.kps,
                               np.zeros_like(self.kds), self.data.qvel[6:], self.kds)
                self.data.ctrl[:] = tau
                
                # Step physics
                mujoco.mj_step(self.model, self.data)
                self.counter += 1
                
                # Update policy
                if self.counter % self.control_decimation == 0:
                    target_pos = self.compute_control()
                
                # Print status
                if time.time() - last_print >= 2.0:
                    pos = self.data.qpos[:3]
                    vel = self.data.qvel[:2]
                    print(f"t={elapsed:5.1f}s | pos=[{pos[0]:5.2f},{pos[1]:5.2f},{pos[2]:5.2f}] | "
                          f"vel=[{vel[0]:5.2f},{vel[1]:5.2f}]m/s")
                    last_print = time.time()
                
                # Sync viewer
                viewer.sync()
                
                # Time keeping
                time_until_next = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)
            
            print(f"\n{'='*70}")
            print(f"Simulation completed: {elapsed:.1f}s")
            print(f"Final position: {self.data.qpos[:3]}")
            print(f"{'='*70}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Deploy B2 policy in MuJoCo')
    parser.add_argument('config', type=str, help='Config file name (in configs/)')
    args = parser.parse_args()
    
    config_path = Path(__file__).parent.parent / 'configs' / args.config
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        return
    
    controller = B2Controller(config_path)
    controller.run()


if __name__ == "__main__":
    main()