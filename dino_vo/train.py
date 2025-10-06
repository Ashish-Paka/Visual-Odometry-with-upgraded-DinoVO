#!/usr/bin/env python3
"""
DINO-VO Training Script

2-Stage Training:
- Stage 1 (4 epochs): Matching loss only
- Stage 2 (10 epochs): Matching + pose loss (λ_D ramps 0.0 → 0.9)

Following Tera.pdf requirements for monocular RGB VO.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
import sys
from tqdm import tqdm
import numpy as np
import os

# Add e2e_multi_view_matching to path
E2E_PATH = Path(__file__).parent.parent / 'e2e_multi_view_matching-master'
sys.path.insert(0, str(E2E_PATH))

from feature_extractor import DINOv3Extractor
from dataloader import create_dataloader
from model import DINOVO
from pose_optimization.two_view.estimate_relative_pose import run_weighted_8_point, run_bundle_adjust_2_view
from pose_optimization.two_view.compute_pose_error import compute_rotation_error, compute_translation_error_as_angle


def normalize_keypoints(kpts, K):
    """Normalize keypoints using camera intrinsics."""
    n_kpts = torch.zeros_like(kpts)
    fx, fy, cx, cy = K[..., 0, 0], K[..., 1, 1], K[..., 0, 2], K[..., 1, 2]
    n_kpts[..., 0] = (kpts[..., 0] - cx.unsqueeze(-1)) / fx.unsqueeze(-1)
    n_kpts[..., 1] = (kpts[..., 1] - cy.unsqueeze(-1)) / fy.unsqueeze(-1)
    return n_kpts


def compute_match_loss(scores, gt_indices, gt_weights):
    """
    Compute matching loss (cross-entropy).

    Args:
        scores: (B, N+1, M+1) score matrix (includes dustbin)
        gt_indices: (B, N) ground truth match indices
        gt_weights: (B, N) confidence weights
    """
    B, N_dustbin, M_dustbin = scores.shape
    N = gt_indices.shape[1]  # Actual number of keypoints (without dustbin)

    # Check for NaN/inf in scores (can happen during training)
    if not torch.isfinite(scores).all():
        print(f"WARNING: Non-finite scores detected in matching loss")
        scores = torch.nan_to_num(scores, nan=0.0, posinf=100.0, neginf=-100.0)

    # Slice to remove dustbin (last row/column)
    # scores_0_1 shape: [B, 197, 197] -> [B, 196, 197]
    # We only need scores for the N real keypoints (not the dustbin row)
    # But keep all M+1 columns (including dustbin) for matching to "no match"
    scores_no_dustbin = scores[:, :N, :]  # (B, N, M+1)

    # Flatten for cross-entropy
    scores_flat = scores_no_dustbin.reshape(-1, M_dustbin)  # (B*N, M+1)
    gt_flat = gt_indices.reshape(-1)  # (B*N,)
    weights_flat = gt_weights.reshape(-1)  # (B*N,)

    # Cross-entropy with weights
    loss = nn.functional.cross_entropy(
        scores_flat,
        gt_flat,
        reduction='none'
    )

    # Check for NaN in loss before weighting
    if not torch.isfinite(loss).all():
        print(f"WARNING: Non-finite loss from cross_entropy")
        loss = torch.nan_to_num(loss, nan=0.0, posinf=100.0, neginf=100.0)

    loss = (loss * weights_flat).sum() / weights_flat.sum().clamp(min=1.0)
    return loss


def compute_gt_matches_from_pose(kpts0, kpts1, K, R_gt, t_gt, threshold=3.0):
    """
    Compute ground truth matches from epipolar geometry.

    Args:
        kpts0, kpts1: (B, N, 2) keypoints in pixel coords
        K: (B, 3, 3) camera intrinsics
        R_gt, t_gt: Ground truth relative pose
        threshold: Epipolar distance threshold in pixels

    Returns:
        gt_indices: (B, N) ground truth match indices (N for matched, dustbin index for unmatched)
        gt_weights: (B, N) confidence weights
    """
    B, N, _ = kpts0.shape
    device = kpts0.device

    # Initialize to dustbin index (N) for unmatched keypoints
    gt_indices = torch.full((B, N), N, dtype=torch.long, device=device)  # Dustbin = index N
    gt_weights = torch.ones((B, N), dtype=torch.float32, device=device)  # All keypoints get weight

    for b in range(B):
        # Essential matrix: E = [t]_x R
        t_cross = torch.tensor([
            [0, -t_gt[b, 2], t_gt[b, 1]],
            [t_gt[b, 2], 0, -t_gt[b, 0]],
            [-t_gt[b, 1], t_gt[b, 0], 0]
        ], device=device, dtype=torch.float32)

        E = t_cross @ R_gt[b]

        # Check for degenerate translation (pure rotation)
        t_norm = torch.norm(t_gt[b])
        if t_norm < 1e-6:
            # Pure rotation - no epipolar constraint, skip matching supervision
            continue

        # Fundamental matrix: F = K^-T E K^-1
        K_inv = torch.inverse(K[b])
        F = K_inv.T @ E @ K_inv

        # Normalize keypoints for matching
        kp0_norm = torch.cat([kpts0[b], torch.ones(N, 1, device=device)], dim=1)  # (N, 3)
        kp1_norm = torch.cat([kpts1[b], torch.ones(N, 1, device=device)], dim=1)  # (M, 3)

        # Compute epipolar lines: l = F @ kp0
        lines = (F @ kp0_norm.T).T  # (N, 3)

        # Distance from kp1 to epipolar lines
        # dist(kp1, l) = |kp1^T @ l| / sqrt(l_x^2 + l_y^2)
        # Add larger epsilon to prevent NaN from degenerate lines
        line_norms = torch.sqrt(lines[:, 0]**2 + lines[:, 1]**2 + 1e-6)
        dists = torch.abs((kp1_norm @ lines.T)) / (line_norms.unsqueeze(0) + 1e-6)  # (M, N)

        # Find nearest neighbors below threshold
        min_dists, nearest = dists.min(dim=0)  # (N,)
        valid = min_dists < threshold

        # Matched keypoints get their nearest neighbor index
        gt_indices[b, valid] = nearest[valid]
        # Unmatched keypoints already initialized to dustbin index N

    return gt_indices, gt_weights


def train_epoch(model, loader, optimizer, epoch, stage, writer, args):
    """Train for one epoch."""
    model.train()

    losses = {
        'total': 0.0,
        'match': 0.0,
        'rot': 0.0,
        'trans': 0.0,
    }

    # Stage 2: Ramp λ_D from 0.0 to 0.9 over 10 epochs
    if stage == 2:
        pose_weight = min(0.9, 0.9 * (epoch - 4) / 6)  # Ramp over epochs 5-10
    else:
        pose_weight = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch} (stage {stage}, λ_D={pose_weight:.3f})")

    for batch_idx, batch in enumerate(pbar):
        # Move to GPU
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        optimizer.zero_grad()

        # Forward pass
        result = model(batch)

        # Compute ground truth matches from pose
        gt_indices, gt_weights = compute_gt_matches_from_pose(
            result['keypoints0'],
            result['keypoints1'],
            batch['K'],
            batch['R_gt'],
            batch['t_gt'],
            threshold=args.match_threshold
        )

        # Matching loss
        match_loss = compute_match_loss(
            result['scores_0_1'],
            gt_indices,
            gt_weights
        )

        # Pose loss (stage 2 only)
        rot_loss = torch.tensor(0.0, device=match_loss.device)
        trans_loss = torch.tensor(0.0, device=match_loss.device)

        if stage == 2:
            # Run weighted 8-point per sample in batch
            B = result['matches0_0_1'].shape[0]
            n_success = 0
            for b in range(B):
                valid = result['matches0_0_1'][b] != -1
                if valid.sum() < 8:
                    continue

                # Prepare per-sample data for pose solver
                pose_data = {
                    'keypoints0': result['keypoints0'][b:b+1],
                    'keypoints1': result['keypoints1'][b:b+1],
                    'intr0': batch['K'][b:b+1],
                    'intr1': batch['K'][b:b+1],
                }

                # Per-sample result dict
                result_single = {
                    'matches0_0_1': result['matches0_0_1'][b:b+1],
                    'matching_scores0_0_1': result['matching_scores0_0_1'][b:b+1],
                    'conf_scores_0_1': result['conf_scores_0_1'][b:b+1] if 'conf_scores_0_1' in result and result['conf_scores_0_1'] is not None else result['matching_scores0_0_1'][b:b+1].unsqueeze(-1),
                }

                # Build target pose (with batch dimension for error computation)
                target = torch.eye(4, device=result['matches0_0_1'].device).unsqueeze(0)  # (1, 4, 4)
                target[0, :3, :3] = batch['R_gt'][b]
                target[0, :3, 3] = batch['t_gt'][b]

                try:
                    # Step 1: Weighted 8-point for initial pose
                    pred_8pt, _ = run_weighted_8_point(
                        pose_data,
                        result_single,
                        id0=0,
                        id1=1,
                        choose_closest=True,   # Use GT to pick correct E-matrix solution
                        target_T_021=target    # Provide ground truth pose
                    )

                    if pred_8pt is not None:
                        # Step 2: Bundle adjustment refinement
                        # Normalize keypoints
                        kpts0_norm = normalize_keypoints(result['keypoints0'][b:b+1], batch['K'][b:b+1])
                        kpts1_norm = normalize_keypoints(result['keypoints1'][b:b+1], batch['K'][b:b+1])

                        # Get confidence (remove last dim if present)
                        conf = result_single['conf_scores_0_1']
                        if conf.dim() == 3:
                            conf = conf.squeeze(-1)

                        # Run bundle adjustment (5 iterations)
                        pred_refined, valid_batch = run_bundle_adjust_2_view(
                            kpts0_norm,
                            kpts1_norm,
                            conf,
                            init_T021=pred_8pt,
                            n_iterations=5
                        )

                        # Use refined pose if bundle adjustment succeeded, else use 8-point result
                        pred = pred_refined if valid_batch[0] else pred_8pt

                        # pred should have shape (1, 4, 4), keep it for error computation
                        # Compute errors (both functions return radians)
                        rot_err_rad = compute_rotation_error(pred, target)
                        trans_err_rad = compute_translation_error_as_angle(pred, target)

                        # Reject outliers (geometric failures - wrong E-matrix solution)
                        if rot_err_rad > torch.pi/4 or trans_err_rad > torch.pi/4:
                            continue  # Skip extreme errors (>45°)

                        # Check for NaN/inf before accumulating
                        if torch.isfinite(rot_err_rad) and torch.isfinite(trans_err_rad):
                            # Normalize to [0, 1] range for stable training
                            rot_loss += (rot_err_rad / torch.pi)  # Now 0-1
                            trans_loss += (trans_err_rad / torch.pi)  # Now 0-1
                            n_success += 1
                except Exception as e:
                    print(f"Warning: Pose estimation failed in batch {b}: {e}")
                    import traceback
                    print(traceback.format_exc())
                    pass

            # Normalize by number of successful pose estimations
            if n_success > 0:
                rot_loss /= n_success
                trans_loss /= n_success

        # Combined loss
        total_loss = (1.0 - pose_weight) * match_loss + pose_weight * (args.rot_weight * rot_loss + args.trans_weight * trans_loss)

        # Check for NaN in total loss before backward
        if not torch.isfinite(total_loss):
            print(f"WARNING: Non-finite total loss detected! match={match_loss.item()}, rot={rot_loss.item()}, trans={trans_loss.item()}")
            print(f"Skipping backward pass for this batch")
            continue

        # Backward
        total_loss.backward()

        # Clip gradients by value BEFORE NaN check (prevents explosion)
        torch.nn.utils.clip_grad_value_(model.trainable_parameters(), 1.0)

        # Check for NaN gradients before stepping
        has_nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                print(f"WARNING: Non-finite gradient in {name}")
                has_nan_grad = True
                break

        if has_nan_grad:
            print("Skipping optimizer step due to NaN gradients")
            optimizer.zero_grad()
            continue

        # Clip gradients by norm for stability
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), args.grad_clip)
        optimizer.step()

        # Track losses
        losses['total'] += total_loss.item()
        losses['match'] += match_loss.item()
        losses['rot'] += rot_loss.item()
        losses['trans'] += trans_loss.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss.item(),
            'match': match_loss.item(),
            'rot': rot_loss.item() if stage == 2 else 0.0,
        })

    # Average losses
    n_batches = len(loader)
    for key in losses:
        losses[key] /= n_batches

    # Log to TensorBoard
    global_step = epoch * len(loader)
    writer.add_scalar('Loss/total', losses['total'], global_step)
    writer.add_scalar('Loss/match', losses['match'], global_step)
    writer.add_scalar('Loss/rot', losses['rot'], global_step)
    writer.add_scalar('Loss/trans', losses['trans'], global_step)
    writer.add_scalar('Train/pose_weight', pose_weight, global_step)

    return losses


def main():
    parser = argparse.ArgumentParser(description='Train DINO-VO')

    # Data
    parser.add_argument('--data_root', type=str, default='data/tartanair')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--num_keypoints', type=int, default=256)
    parser.add_argument('--max_samples', type=int, default=None, help='Limit dataset size (for fast PoC training)')

    # Training
    parser.add_argument('--epochs', type=int, default=14)
    parser.add_argument('--stage1_epochs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # Loss weights
    parser.add_argument('--match_threshold', type=float, default=3.0)
    parser.add_argument('--rot_weight', type=float, default=1.0)  # Errors now in degrees (0-180)
    parser.add_argument('--trans_weight', type=float, default=1.0)  # Errors now in degrees (0-180)

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--save_every', type=int, default=2)
    parser.add_argument('--exp_name', type=str, default='dinovo_tartanair')

    args = parser.parse_args()

    # Setup
    checkpoint_dir = Path(args.checkpoint_dir) / args.exp_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=checkpoint_dir / 'logs')

    print("=" * 70)
    print("DINO-VO Training")
    print("=" * 70)
    print(f"Experiment: {args.exp_name}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Epochs: {args.epochs} (Stage 1: {args.stage1_epochs}, Stage 2: {args.epochs - args.stage1_epochs})")
    print("=" * 70)

    # Create dataloader
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"Error: Data not found at {data_root}")
        print("Please run download_tartanair.py first")
        return

    # Create train loader
    os.environ['DATASET_SPLIT'] = 'train'
    train_loader = create_dataloader(
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        image_size=(args.image_size, args.image_size),
        max_samples=args.max_samples,
    )

    # Create validation loader
    os.environ['DATASET_SPLIT'] = 'val'
    val_loader = create_dataloader(
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        image_size=(args.image_size, args.image_size),
        max_samples=None,  # Use all validation data
    )

    print(f"\n✓ Train Dataset: {len(train_loader.dataset)} frame pairs")
    print(f"✓ Val Dataset: {len(val_loader.dataset)} frame pairs")
    print(f"✓ Train Batches per epoch: {len(train_loader)}")

    # Use train_loader as main loader (keep variable name for compatibility)
    loader = train_loader

    # Create model
    model = DINOVO(
        num_keypoints=args.num_keypoints,
        image_size=args.image_size,
    ).cuda()

    # Optimizer (only trainable parameters)
    optimizer = optim.Adam(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Training loop
    print(f"\n{'=' * 70}")
    print("Starting training...")
    print(f"{'=' * 70}\n")

    for epoch in range(1, args.epochs + 1):
        stage = 1 if epoch <= args.stage1_epochs else 2

        losses = train_epoch(model, loader, optimizer, epoch, stage, writer, args)

        print(f"\nEpoch {epoch}/{args.epochs} - Stage {stage}")
        print(f"  Total Loss: {losses['total']:.4f}")
        print(f"  Match Loss: {losses['match']:.4f}")
        if stage == 2:
            print(f"  Rot Loss: {losses['rot']:.4f}")
            print(f"  Trans Loss: {losses['trans']:.4f}")

        # Save checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = checkpoint_dir / f'epoch_{epoch:03d}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': losses,
            }, ckpt_path)
            print(f"  ✓ Saved checkpoint: {ckpt_path}")

    writer.close()

    print(f"\n{'=' * 70}")
    print("✓ Training complete!")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
