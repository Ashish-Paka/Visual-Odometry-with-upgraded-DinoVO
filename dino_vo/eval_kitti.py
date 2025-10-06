#!/usr/bin/env python3
"""
KITTI Evaluation for DINO-VO

Evaluate trained model on KITTI sequences and compute:
- ATE (Absolute Trajectory Error)
- RPE (Relative Pose Error)
- Generate trajectory visualizations (similar to DINO-VO Figure 6)
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

E2E_PATH = Path(__file__).parent.parent / 'e2e_multi_view_matching-master'
sys.path.insert(0, str(E2E_PATH))

from model import DINOVO
from pose_optimization.two_view.estimate_relative_pose import run_weighted_8_point, run_bundle_adjust_2_view
import cv2


def normalize_keypoints(kpts, K):
    """Normalize keypoints using camera intrinsics."""
    n_kpts = torch.zeros_like(kpts)
    fx, fy, cx, cy = K[..., 0, 0], K[..., 1, 1], K[..., 0, 2], K[..., 1, 2]
    n_kpts[..., 0] = (kpts[..., 0] - cx.unsqueeze(-1)) / fx.unsqueeze(-1)
    n_kpts[..., 1] = (kpts[..., 1] - cy.unsqueeze(-1)) / fy.unsqueeze(-1)
    return n_kpts


class KITTISequence:
    """Load KITTI odometry sequence."""

    def __init__(self, sequence_dir, sequence_id):
        self.sequence_dir = Path(sequence_dir)
        self.sequence_id = sequence_id

        # Paths
        self.image_dir = self.sequence_dir / 'sequences' / sequence_id / 'image_0'
        self.pose_file = self.sequence_dir / 'poses' / f'{sequence_id}.txt'
        self.calib_file = self.sequence_dir / 'sequences' / sequence_id / 'calib.txt'

        # Load
        self.images = sorted(self.image_dir.glob('*.png'))
        self.poses_gt = self._load_poses()
        self.K = self._load_calibration()

        print(f"✓ KITTI Sequence {sequence_id}: {len(self.images)} frames")

    def _load_poses(self):
        """Load ground truth poses (3x4 projection matrices)."""
        poses = []
        with open(self.pose_file, 'r') as f:
            for line in f:
                T = np.array([float(x) for x in line.strip().split()]).reshape(3, 4)
                # Convert to 4x4
                T_full = np.eye(4)
                T_full[:3, :] = T
                poses.append(T_full)
        return poses

    def _load_calibration(self):
        """Load camera intrinsics."""
        with open(self.calib_file, 'r') as f:
            for line in f:
                if line.startswith('P0:'):
                    P0 = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 4)
                    K = P0[:3, :3]
                    return K
        return np.eye(3)

    def __len__(self):
        return len(self.images)

    def get_pair(self, idx):
        """Get consecutive frame pair."""
        img0 = cv2.imread(str(self.images[idx]))
        img1 = cv2.imread(str(self.images[idx + 1]))

        # Convert to RGB
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        return img0, img1


def preprocess_image(img, size=224):
    """Preprocess image for model."""
    img_resized = cv2.resize(img, (size, size))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    return img_tensor.unsqueeze(0)


def estimate_pose_from_matches(result, K0, K1, debug=False):
    """Estimate pose from matches using weighted 8-point."""
    # Check matches
    matches = result['matches0_0_1']
    valid_matches = (matches[0] != -1).sum().item() if matches.dim() > 1 else (matches != -1).sum().item()

    if debug:
        print(f"[DEBUG] Valid matches: {valid_matches}")
        print(f"[DEBUG] matches0_0_1 shape: {result['matches0_0_1'].shape}")

    # Prepare data for pose solver (e2e expects 'intr0', 'intr1')
    pose_data = {
        'keypoints0': result['keypoints0'],
        'keypoints1': result['keypoints1'],
        'intr0': K0,
        'intr1': K1,
    }

    # CRITICAL FIX: Ensure conf_scores exists and is not None
    # The checkpoint was trained without conf_mlp, so we use matching_scores as confidence
    if 'conf_scores_0_1' not in result or result['conf_scores_0_1'] is None:
        # matching_scores0_0_1 is (B, N), we need (B, N, 1) for confidence
        result['conf_scores_0_1'] = result['matching_scores0_0_1'].unsqueeze(-1)

    try:
        # Step 1: Run weighted 8-point for initial pose
        # NOTE: choose_closest=False because we don't have ground truth in evaluation
        # The e2e code will use cheirality check to pick the correct solution
        pred_T_8pt, _ = run_weighted_8_point(
            pose_data,
            result,
            id0=0,
            id1=1,
            choose_closest=False,
        )
        if pred_T_8pt is None:
            if debug:
                print(f"[DEBUG] run_weighted_8_point returned None!")
            return torch.eye(4, device=K0.device)

        # Step 2: Bundle adjustment refinement
        kpts0_norm = normalize_keypoints(result['keypoints0'], K0)
        kpts1_norm = normalize_keypoints(result['keypoints1'], K1)

        # Get confidence
        conf = result['conf_scores_0_1']
        if conf.dim() == 3:
            conf = conf.squeeze(-1)

        # Run bundle adjustment
        pred_T_refined, valid_batch = run_bundle_adjust_2_view(
            kpts0_norm,
            kpts1_norm,
            conf,
            init_T021=pred_T_8pt,
            n_iterations=5
        )

        # Use refined pose if bundle adjustment succeeded
        pred_T = pred_T_refined if valid_batch[0] else pred_T_8pt

        return pred_T
    except Exception as e:
        import traceback
        print(f"Warning: Pose estimation failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return torch.eye(4, device=K0.device)


def align_trajectory_scale(poses_pred, poses_gt):
    """
    Align predicted trajectory to ground truth using scale factor.
    Essential for monocular VO since scale is unobservable.

    Returns aligned predicted poses and scale factor.
    """
    # Compute relative translations
    pred_rel_norms = []
    gt_rel_norms = []

    for i in range(min(len(poses_pred), len(poses_gt)) - 1):
        # Relative translation magnitudes
        pred_t_rel = poses_pred[i+1][:3, 3] - poses_pred[i][:3, 3]
        gt_t_rel = poses_gt[i+1][:3, 3] - poses_gt[i][:3, 3]

        pred_rel_norms.append(np.linalg.norm(pred_t_rel))
        gt_rel_norms.append(np.linalg.norm(gt_t_rel))

    pred_rel_norms = np.array(pred_rel_norms)
    gt_rel_norms = np.array(gt_rel_norms)

    # Least squares scale: s = (pred · gt) / (pred · pred)
    scale = np.sum(pred_rel_norms * gt_rel_norms) / (np.sum(pred_rel_norms ** 2) + 1e-8)

    # Apply scale to all predicted poses
    poses_aligned = []
    for pose in poses_pred:
        pose_scaled = pose.copy()
        pose_scaled[:3, 3] *= scale
        poses_aligned.append(pose_scaled)

    return poses_aligned, scale


def compute_ate(poses_pred, poses_gt):
    """
    Compute Absolute Trajectory Error with scale alignment.
    Standard metric for monocular VO evaluation.
    """
    # Align scale first
    poses_aligned, scale = align_trajectory_scale(poses_pred, poses_gt)

    errors = []
    for i in range(min(len(poses_aligned), len(poses_gt))):
        pred_pos = poses_aligned[i][:3, 3]
        gt_pos = poses_gt[i][:3, 3]
        error = np.linalg.norm(pred_pos - gt_pos)
        errors.append(error)

    return np.array(errors), scale


def compute_rpe(poses_pred, poses_gt, delta=1):
    """Compute Relative Pose Error."""
    rot_errors = []
    trans_errors = []

    for i in range(len(poses_pred) - delta):
        # Relative transformations
        pred_rel = np.linalg.inv(poses_pred[i]) @ poses_pred[i + delta]
        gt_rel = np.linalg.inv(poses_gt[i]) @ poses_gt[i + delta]

        # Rotation error
        R_error = np.linalg.inv(pred_rel[:3, :3]) @ gt_rel[:3, :3]
        angle_error = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))
        rot_errors.append(np.degrees(angle_error))

        # Translation error
        t_pred = pred_rel[:3, 3]
        t_gt = gt_rel[:3, 3]
        trans_error = np.linalg.norm(t_pred - t_gt) / np.linalg.norm(t_gt) * 100  # percent
        trans_errors.append(trans_error)

    return np.array(rot_errors), np.array(trans_errors)


def plot_trajectory(poses_pred, poses_gt, sequence_id, save_path, scale_factor=1.0):
    """Plot 2D trajectory (bird's eye view) with scale alignment."""
    # Align scale
    poses_aligned, _ = align_trajectory_scale(poses_pred, poses_gt)

    pred_traj = np.array([p[:3, 3] for p in poses_aligned])
    gt_traj = np.array([p[:3, 3] for p in poses_gt[:len(poses_aligned)]])

    plt.figure(figsize=(10, 10))
    plt.plot(gt_traj[:, 0], gt_traj[:, 2], 'b-', label='Ground Truth', linewidth=2)
    plt.plot(pred_traj[:, 0], pred_traj[:, 2], 'r--', label=f'DINO-VO (scale={scale_factor:.3f})', linewidth=2)
    plt.xlabel('x (m)', fontsize=14)
    plt.ylabel('z (m)', fontsize=14)
    plt.title(f'KITTI Sequence {sequence_id} Trajectory', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved trajectory plot: {save_path}")


def evaluate_sequence(model, sequence, device, image_size=224):
    """Evaluate model on a KITTI sequence."""
    model.eval()

    # Initialize trajectory with first pose
    poses_pred = [sequence.poses_gt[0]]

    print(f"\nProcessing {len(sequence) - 1} frame pairs...")

    with torch.no_grad():
        for i in tqdm(range(len(sequence) - 1)):
            # Get frame pair
            img0, img1 = sequence.get_pair(i)

            # Preprocess
            img0_tensor = preprocess_image(img0, image_size).to(device)
            img1_tensor = preprocess_image(img1, image_size).to(device)

            # Prepare batch
            K_tensor = torch.from_numpy(sequence.K).float().unsqueeze(0).to(device)

            batch = {
                'image0': img0_tensor,
                'image1': img1_tensor,
                'K': K_tensor,
            }

            # Forward pass
            result = model(batch)

            # Estimate relative pose (debug first 3 frames)
            debug = (i < 3)
            pred_rel = estimate_pose_from_matches(result, K_tensor, K_tensor, debug=debug)

            # Remove batch dimension if present
            if pred_rel.dim() == 3:
                pred_rel = pred_rel[0]

            # Debug: print transformation details for first few frames
            if i < 3:
                print(f"\n[Frame {i}->{i+1}]")
                print(f"Predicted T_rel:\n{pred_rel.cpu().numpy()}")
                print(f"Translation: {pred_rel[:3, 3].cpu().numpy()}")
                print(f"Translation norm: {torch.norm(pred_rel[:3, 3]).item():.4f}")

                # Compute GT relative pose for comparison
                T_gt_0 = sequence.poses_gt[i]
                T_gt_1 = sequence.poses_gt[i+1]
                T_gt_rel = np.linalg.inv(T_gt_0) @ T_gt_1
                print(f"GT T_rel:\n{T_gt_rel}")
                print(f"GT translation: {T_gt_rel[:3, 3]}")
                print(f"GT translation norm: {np.linalg.norm(T_gt_rel[:3, 3]):.4f}")

            # Compose absolute pose
            pred_abs = poses_pred[-1] @ pred_rel.cpu().numpy()
            poses_pred.append(pred_abs)

    return poses_pred


def main():
    parser = argparse.ArgumentParser(description='Evaluate DINO-VO on KITTI')

    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--kitti_dir', type=str, default='data/kitti', help='KITTI data directory')
    parser.add_argument('--sequences', type=str, nargs='+', default=['00', '05', '06', '07'])
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--results_dir', type=str, default='results/kitti')

    args = parser.parse_args()

    # Setup
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("DINO-VO KITTI Evaluation")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Sequences: {args.sequences}")
    print(f"Device: {device}")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model = DINOVO(image_size=args.image_size).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Evaluate each sequence
    all_metrics = {}

    for seq_id in args.sequences:
        print(f"\n{'='*70}")
        print(f"Evaluating Sequence {seq_id}")
        print(f"{'='*70}")

        # Load sequence
        sequence = KITTISequence(args.kitti_dir, seq_id)

        # Run evaluation
        poses_pred = evaluate_sequence(model, sequence, device, args.image_size)

        # Compute metrics with scale alignment
        ate, scale_factor = compute_ate(poses_pred, sequence.poses_gt)
        rpe_rot, rpe_trans = compute_rpe(poses_pred, sequence.poses_gt, delta=10)

        # Print metrics
        print(f"\n✓ Sequence {seq_id} Results:")
        print(f"  Scale factor: {scale_factor:.4f}")
        print(f"  ATE (mean): {ate.mean():.3f} m")
        print(f"  ATE (std):  {ate.std():.3f} m")
        print(f"  RPE Rotation (mean): {rpe_rot.mean():.3f}°/10frames")
        print(f"  RPE Translation (mean): {rpe_trans.mean():.3f}%")

        # Save metrics
        all_metrics[seq_id] = {
            'scale_factor': float(scale_factor),
            'ate_mean': float(ate.mean()),
            'ate_std': float(ate.std()),
            'rpe_rot_mean': float(rpe_rot.mean()),
            'rpe_trans_mean': float(rpe_trans.mean()),
        }

        # Plot trajectory with scale alignment
        traj_path = results_dir / f'trajectory_seq{seq_id}.png'
        plot_trajectory(poses_pred, sequence.poses_gt, seq_id, traj_path, scale_factor)

        # Save scale-aligned predicted trajectory
        poses_aligned, _ = align_trajectory_scale(poses_pred, sequence.poses_gt)
        traj_file = results_dir / f'poses_seq{seq_id}.txt'
        with open(traj_file, 'w') as f:
            for pose in poses_aligned:
                f.write(' '.join(map(str, pose[:3, :].flatten())) + '\n')
        print(f"  ✓ Saved scale-aligned poses: {traj_file}")

    # Summary
    print(f"\n{'='*70}")
    print("Evaluation Summary")
    print(f"{'='*70}")

    for seq_id, metrics in all_metrics.items():
        print(f"\nSequence {seq_id}:")
        print(f"  Scale: {metrics['scale_factor']:.4f}")
        print(f"  ATE: {metrics['ate_mean']:.3f} ± {metrics['ate_std']:.3f} m")
        print(f"  RPE Rot: {metrics['rpe_rot_mean']:.3f}°/10frames")
        print(f"  RPE Trans: {metrics['rpe_trans_mean']:.3f}%")

    print(f"\n✓ Results saved to: {results_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
