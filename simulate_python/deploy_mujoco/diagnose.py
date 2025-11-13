"""Diagnose observation values and policy outputs."""
import numpy as np
import mujoco
import torch
import yaml
from pathlib import Path
from b2_flat_locomotion import B2Controller, quaternion_to_gravity

class DiagnosticController(B2Controller):
    """Extended controller with diagnostics."""
    
    def __init__(self, config_path):
        super().__init__(config_path)
        self.step_count = 0
    
    def run_diagnostic(self, num_steps=10):
        """Run diagnostic for first few steps."""
        self.reset()
        
        print("\n" + "="*70)
        print("DIAGNOSTIC MODE: First 10 control steps")
        print("="*70 + "\n")
        
        target_pos = self.default_angles_mj.copy()
        
        for step in range(num_steps * self.control_decimation):
            # PD control
            tau = (target_pos - self.data.qpos[7:]) * self.kps - self.data.qvel[6:] * self.kds
            tau = np.clip(tau, -150, 150)  # ✓ Add torque limiting
            self.data.ctrl[:] = tau
            
            # Step physics
            mujoco.mj_step(self.model, self.data)
            self.counter += 1
            
            # Update policy and print diagnostics
            if self.counter % self.control_decimation == 0:
                self.step_count += 1
                
                print(f"\n{'─'*70}")
                print(f"Step {self.step_count}")
                print(f"{'─'*70}")
                
                # Get raw observation components
                quat = self.data.qpos[3:7]
                ang_vel = self.data.qvel[3:6]
                gravity = quaternion_to_gravity(quat)
                
                joint_pos_mj = self.data.qpos[7:]
                joint_vel_mj = self.data.qvel[6:]
                joint_pos = joint_pos_mj[self.policy_to_mujoco]
                joint_vel = joint_vel_mj[self.policy_to_mujoco]
                
                print(f"\n1. Base state:")
                print(f"   Position: {self.data.qpos[:3]}")
                print(f"   Quaternion: {quat}")
                print(f"   Angular vel: {ang_vel}")
                print(f"   Gravity (body): {gravity}")
                
                print(f"\n2. Joint positions (policy order, first 6):")
                print(f"   Absolute: {joint_pos[:6]}")
                print(f"   Relative: {(joint_pos - self.default_angles)[:6]}")
                
                print(f"\n3. Joint velocities (policy order, first 6):")
                print(f"   {joint_vel[:6]}")
                
                # Get observation (before normalization)
                obs_raw = self.get_observation_raw()
                
                print(f"\n4. Observation (before normalization):")
                print(f"   [0:3]   ang_vel:  {obs_raw[0:3]}")
                print(f"   [3:6]   gravity:  {obs_raw[3:6]}")
                print(f"   [6:9]   cmd:      {obs_raw[6:9]}")
                print(f"   [9:12]  joint_pos (first 3): {obs_raw[9:12]}")
                print(f"   [21:24] joint_vel (first 3): {obs_raw[21:24]}")
                print(f"   [33:36] last_action (first 3): {obs_raw[33:36]}")
                
                # Get normalized observation
                obs = self.get_observation()
                
                print(f"\n5. Observation (after normalization):")
                print(f"   [0:3]   ang_vel:  {obs[0:3]}")
                print(f"   [3:6]   gravity:  {obs[3:6]}")
                print(f"   [6:9]   cmd:      {obs[6:9]}")
                print(f"   Range: [{obs.min():.3f}, {obs.max():.3f}]")
                
                # Policy inference (WITH CLIPPING - this is the fix!)
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
                    action_tensor = self.policy(obs_tensor)
                    action_raw = action_tensor.squeeze(0).numpy()
                
                print(f"\n6. Policy output (BEFORE clipping):")
                print(f"   Raw action: {action_raw[:6]}")
                print(f"   Range: [{action_raw.min():.3f}, {action_raw.max():.3f}]")
                
                # ✓ APPLY CLIPPING (this matches your b2_flat_locomotion.py)
                action = np.clip(action_raw, -2.0, 2.0)
                self.action = action.astype(np.float32)
                
                print(f"\n7. Policy output (AFTER clipping to [-2, 2]):")
                print(f"   Clipped action: {action[:6]}")
                print(f"   Range: [{action.min():.3f}, {action.max():.3f}]")
                
                # Convert to target positions
                target_pos_policy = self.action * self.action_scale + self.default_angles
                
                print(f"\n8. Target positions:")
                print(f"   Policy order (first 6): {target_pos_policy[:6]}")
                print(f"   Change from default: {(target_pos_policy - self.default_angles)[:6]}")
                print(f"   Max change: {np.abs(target_pos_policy - self.default_angles).max():.3f} rad")
                
                # Update target
                target_pos = np.zeros(self.num_actions, dtype=np.float32)
                for pol_idx in range(self.num_actions):
                    mj_idx = self.policy_to_mujoco[pol_idx]
                    target_pos[mj_idx] = target_pos_policy[pol_idx]
                
                print(f"\n9. Control torques (first 6):")
                print(f"   {tau[:6]}")
                print(f"   Max torque: {np.abs(tau).max():.2f} Nm")
                
                if self.step_count >= num_steps:
                    break
        
        print("\n" + "="*70)
        print("Diagnostic complete")
        print("="*70)
    
    def get_observation_raw(self):
        """Get observation without normalization."""
        quat = self.data.qpos[3:7]
        ang_vel = self.data.qvel[3:6]
        gravity = quaternion_to_gravity(quat)
        
        joint_pos_mj = self.data.qpos[7:]
        joint_vel_mj = self.data.qvel[6:]
        joint_pos = joint_pos_mj[self.policy_to_mujoco]
        joint_vel = joint_vel_mj[self.policy_to_mujoco]
        
        joint_pos_rel = (joint_pos - self.default_angles) * self.dof_pos_scale
        joint_vel_scaled = joint_vel * self.dof_vel_scale
        ang_vel_scaled = ang_vel * self.ang_vel_scale
        
        obs = np.zeros(self.num_obs, dtype=np.float32)
        obs[0:3] = ang_vel_scaled
        obs[3:6] = gravity
        obs[6:9] = self.cmd * self.cmd_scale
        obs[9:21] = joint_pos_rel
        obs[21:33] = joint_vel_scaled
        obs[33:45] = self.action
        
        return obs


def main():
    config_path = Path(__file__).parent.parent / 'configs' / 'b2_flat_locomotion.yaml'
    controller = DiagnosticController(config_path)
    controller.run_diagnostic(num_steps=5)


if __name__ == "__main__":
    main()