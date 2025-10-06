#!/usr/bin/env python3
"""
Test pose convention and composition to identify errors.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add e2e path
E2E_PATH = Path(__file__).parent / 'e2e_multi_view_matching-master'
sys.path.insert(0, str(E2E_PATH))

from kornia.geometry.epipolar import motion_from_essential_choose_solution

def test_essential_matrix_convention():
    """
    Test what convention motion_from_essential uses.

    Create two camera poses, compute Essential matrix, decompose it,
    and check if we get back the correct relative transformation.
    """
    print("="*70)
    print("Testing Essential Matrix Convention")
    print("="*70)

    # Camera intrinsics (identity for normalized coords)
    K = torch.eye(3).unsqueeze(0)

    # Camera 0 at origin
    T0 = np.eye(4)

    # Camera 1 translated +1 in x direction
    T1 = np.eye(4)
    T1[0, 3] = 1.0

    # Ground truth relative pose: T_0_to_1 = inv(T0) @ T1
    T_rel_gt = np.linalg.inv(T0) @ T1
    R_gt = T_rel_gt[:3, :3]
    t_gt = T_rel_gt[:3, 3]

    print(f"\nGround Truth Relative Pose (T_0_to_1):")
    print(f"R_gt:\n{R_gt}")
    print(f"t_gt: {t_gt}")

    # Compute Essential matrix: E = [t]_x @ R
    t_cross = np.array([
        [0, -t_gt[2], t_gt[1]],
        [t_gt[2], 0, -t_gt[0]],
        [-t_gt[1], t_gt[0], 0]
    ])
    E = t_cross @ R_gt

    print(f"\nEssential Matrix E:")
    print(E)

    # Convert to torch
    E_torch = torch.from_numpy(E).float().unsqueeze(0)

    # Create synthetic correspondences (points in normalized coordinates)
    # Points in camera 0
    pts0 = torch.tensor([[
        [0.1, 0.1],
        [0.5, 0.2],
        [-0.3, 0.4],
        [0.0, 0.0],
        [0.6, -0.2],
        [0.8, 0.1],
        [-0.5, -0.5],
        [0.3, 0.7]
    ]], dtype=torch.float32)

    # Transform to camera 1 using ground truth
    pts0_homo = torch.cat([pts0, torch.ones(1, 8, 1)], dim=-1)  # Add z=1

    # Assuming depth = 5.0 for all points
    pts_3d_cam0 = pts0_homo * 5.0  # (1, 8, 3)

    # Transform to camera 1
    R_torch = torch.from_numpy(R_gt).float().unsqueeze(0)
    t_torch = torch.from_numpy(t_gt).float().unsqueeze(0).unsqueeze(-1)

    pts_3d_cam1 = (R_torch @ pts_3d_cam0.transpose(1, 2)).transpose(1, 2) + t_torch.transpose(1, 2)

    # Project back to normalized coordinates
    pts1 = pts_3d_cam1[:, :, :2] / pts_3d_cam1[:, :, 2:3]

    print(f"\nSynthetic correspondences:")
    print(f"pts0: {pts0[0, :3]}")  # Show first 3 points
    print(f"pts1: {pts1[0, :3]}")

    # Decompose Essential matrix
    R_recovered, t_recovered, pts_3d = motion_from_essential_choose_solution(
        E_torch, K, K, pts0, pts1, mask=None
    )

    print(f"\nRecovered from Essential Matrix:")
    print(f"R_recovered:\n{R_recovered[0].numpy()}")
    print(f"t_recovered: {t_recovered[0, :, 0].numpy()}")

    # Check if recovered matches ground truth
    R_match = np.allclose(R_recovered[0].numpy(), R_gt, atol=1e-3)

    # Translation is up to scale, check direction
    t_recovered_np = t_recovered[0, :, 0].numpy()
    t_scale = np.linalg.norm(t_recovered_np)
    t_recovered_normalized = t_recovered_np / t_scale if t_scale > 1e-6 else t_recovered_np

    t_gt_scale = np.linalg.norm(t_gt)
    t_gt_normalized = t_gt / t_gt_scale if t_gt_scale > 1e-6 else t_gt

    t_match = np.allclose(t_recovered_normalized, t_gt_normalized, atol=1e-3)

    print(f"\nValidation:")
    print(f"  Rotation matches GT: {R_match}")
    print(f"  Translation direction matches GT: {t_match}")
    print(f"  t_gt normalized: {t_gt_normalized}")
    print(f"  t_recovered normalized: {t_recovered_normalized}")

    if R_match and t_match:
        print("\n✓ Essential matrix decomposes to T_0_to_1 (transform from cam0 to cam1)")
        print("  This means: p_cam1 = R @ p_cam0 + t")
    else:
        print("\n✗ Convention mismatch detected!")

    print("\n" + "="*70)

    return R_match and t_match


def test_pose_accumulation():
    """
    Test pose accumulation logic used in evaluation.
    """
    print("\nTesting Pose Accumulation")
    print("="*70)

    # Simulate ground truth trajectory
    T_world_to_cam0 = np.eye(4)
    T_world_to_cam1 = np.eye(4)
    T_world_to_cam1[0, 3] = 1.0  # Move +1 in x
    T_world_to_cam2 = np.eye(4)
    T_world_to_cam2[0, 3] = 2.0  # Move +2 in x

    poses_gt = [T_world_to_cam0, T_world_to_cam1, T_world_to_cam2]

    print("Ground Truth Trajectory:")
    for i, T in enumerate(poses_gt):
        print(f"  Cam {i} position: {T[:3, 3]}")

    # Simulate predicted relative poses
    T_rel_0_to_1 = np.linalg.inv(T_world_to_cam0) @ T_world_to_cam1
    T_rel_1_to_2 = np.linalg.inv(T_world_to_cam1) @ T_world_to_cam2

    print(f"\nRelative Poses (GT):")
    print(f"  T_0_to_1 translation: {T_rel_0_to_1[:3, 3]}")
    print(f"  T_1_to_2 translation: {T_rel_1_to_2[:3, 3]}")

    # Accumulate using eval code logic: poses_pred[-1] @ pred_rel
    poses_pred = [T_world_to_cam0.copy()]  # Start from GT first pose

    poses_pred.append(poses_pred[-1] @ T_rel_0_to_1)
    poses_pred.append(poses_pred[-1] @ T_rel_1_to_2)

    print(f"\nAccumulated Predicted Trajectory:")
    for i, T in enumerate(poses_pred):
        print(f"  Cam {i} position: {T[:3, 3]}")

    # Check if accumulated matches GT
    match = all(np.allclose(poses_pred[i][:3, 3], poses_gt[i][:3, 3]) for i in range(3))

    if match:
        print("\n✓ Pose accumulation logic is CORRECT")
        print("  Formula: T_world_to_cam_{i+1} = T_world_to_cam_i @ T_cam_i_to_cam_{i+1}")
    else:
        print("\n✗ Pose accumulation logic is WRONG!")
        print("  Mismatch between accumulated and ground truth")

    print("="*70)

    return match


if __name__ == '__main__':
    test1_pass = test_essential_matrix_convention()
    test2_pass = test_pose_accumulation()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Essential Matrix Convention Test: {'✓ PASS' if test1_pass else '✗ FAIL'}")
    print(f"Pose Accumulation Test: {'✓ PASS' if test2_pass else '✗ FAIL'}")
    print("="*70)
