"""Extract observation normalization parameters from RSL-RL checkpoint."""
import torch
import numpy as np
from pathlib import Path
import argparse

"""
Command to run: python3 export_normalization.py ../exported_policy/b2_flat_locomotion/model_.pt 
--output-dir ../exported_policy/b2_flat_locomotion
"""


def extract_normalization_from_checkpoint(checkpoint_path, output_dir):
    """
    Extract obs_mean and obs_std from RSL-RL checkpoint with EmpiricalNormalization.
    
    Args:
        checkpoint_path: Path to model_*.pt checkpoint file
        output_dir: Directory to save obs_mean.npy and obs_std.npy
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"\nCheckpoint top-level keys:")
    for key in checkpoint.keys():
        print(f"  - {key}")
    
    # Get model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    print(f"\nSearching for EmpiricalNormalization parameters...")
    print(f"Total keys in state_dict: {len(state_dict)}")
    
    obs_mean = None
    obs_std = None
    
    # First try direct actor normalization keys
    direct_keys = [
        ('actor_obs_normalizer._mean', 'actor_obs_normalizer._std'),
        ('actor.obs_normalizer._mean', 'actor.obs_normalizer._std'),
        ('actor.normalizer._mean', 'actor.normalizer._std'),
    ]
    
    print(f"\nTrying direct actor keys:")
    for mean_key, std_key in direct_keys:
        print(f"  Trying: {mean_key}")
        if mean_key in state_dict and std_key in state_dict:
            obs_mean = state_dict[mean_key].cpu().numpy()
            obs_std = state_dict[std_key].cpu().numpy()
            print(f"  ✓ Found: {mean_key}, {std_key}")
            break
    
    # If not found, search systematically
    if obs_mean is None:
        print(f"\n⚠️  Direct keys not found. Searching systematically...")
        mean_keys = [k for k in state_dict.keys() if 'mean' in k.lower()]
        std_keys = [k for k in state_dict.keys() if 'std' in k.lower()]
        
        print(f"\nKeys containing 'mean' ({len(mean_keys)}):")
        for key in mean_keys:
            print(f"  - {key}: {state_dict[key].shape}")
        
        print(f"\nKeys containing 'std' ({len(std_keys)}):")
        for key in std_keys:
            print(f"  - {key}: {state_dict[key].shape}")
        
        # Find actor keys specifically
        actor_mean_keys = [k for k in mean_keys if 'actor' in k.lower() and 'critic' not in k.lower()]
        
        if actor_mean_keys:
            mean_key = actor_mean_keys[0]
            # Find corresponding std key
            base_key = mean_key.replace('._mean', '')
            std_key = f'{base_key}._std'
            
            if std_key in state_dict:
                obs_mean = state_dict[mean_key].cpu().numpy()
                obs_std = state_dict[std_key].cpu().numpy()
                print(f"\n✓ Found actor normalization:")
                print(f"  Mean: {mean_key}")
                print(f"  Std:  {std_key}")
            else:
                print(f"\n❌ Found mean but not std: {mean_key}")
        else:
            print(f"\n❌ No actor normalization keys found!")
    
    if obs_mean is None:
        print("\n❌ ERROR: Could not find normalization parameters!")
        return False
    
    # Remove batch dimension if present [1, obs_dim] -> [obs_dim]
    if len(obs_mean.shape) == 2 and obs_mean.shape[0] == 1:
        obs_mean = obs_mean.squeeze(0)
    if len(obs_std.shape) == 2 and obs_std.shape[0] == 1:
        obs_std = obs_std.squeeze(0)
    
    # Validate
    print(f"\n{'='*70}")
    print("Extracted Normalization Parameters")
    print(f"{'='*70}")
    print(f"  obs_mean shape: {obs_mean.shape}")
    print(f"  obs_std shape:  {obs_std.shape}")
    print(f"  Mean range: [{obs_mean.min():.6f}, {obs_mean.max():.6f}]")
    print(f"  Std range:  [{obs_std.min():.6f}, {obs_std.max():.6f}]")
    
    # Show first few values
    print(f"\n  First 9 mean values:")
    print(f"    {obs_mean[:9]}")
    print(f"  First 9 std values:")
    print(f"    {obs_std[:9]}")
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mean_path = output_dir / 'obs_mean.npy'
    std_path = output_dir / 'obs_std.npy'
    
    np.save(mean_path, obs_mean.astype(np.float32))
    np.save(std_path, obs_std.astype(np.float32))
    
    print(f"\n✓ Saved normalization parameters:")
    print(f"  {mean_path}")
    print(f"  {std_path}")
    print(f"{'='*70}\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Extract observation normalization from RSL-RL checkpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from checkpoint
  python export_normalization.py /path/to/model_2000.pt
  
  # Specify custom output directory
  python export_normalization.py /path/to/model_2000.pt --output-dir ./my_policy
        """
    )
    parser.add_argument('checkpoint', type=str, help='Path to model_*.pt checkpoint file')
    parser.add_argument('--output-dir', type=str, 
                       default='../exported_policy/b2_flat_locomotion',
                       help='Output directory for obs_mean.npy and obs_std.npy')
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found: {checkpoint_path}")
        return
    
    success = extract_normalization_from_checkpoint(checkpoint_path, args.output_dir)
    
    if success:
        print("\n" + "="*70)
        print("✓ SUCCESS! Normalization parameters extracted.")
        print("  You can now run simulation with normalization enabled.")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("❌ FAILED! Could not extract normalization parameters.")
        print("  Options:")
        print("  1. Check if checkpoint is correct")
        print("  2. Manually inspect checkpoint and update script")
        print("  3. Disable normalization in config (use_obs_normalization: false)")
        print("="*70)


if __name__ == "__main__":
    main()