#!/usr/bin/env python3
"""
TartanAir Dataloader for Monocular RGB Visual Odometry

Loads consecutive RGB frame pairs with ground truth poses.
No depth required - pure monocular VO.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import os


class TartanAirVO(Dataset):
    """
    TartanAir dataset for monocular RGB visual odometry.

    Returns consecutive frame pairs with relative pose transformations.
    """

    def __init__(
        self,
        data_root,
        image_size=(224, 224),
        sequence_length=2,
        environments=None,
        trajectories=None,
        max_samples=None,  # Limit dataset size for faster training
    ):
        """
        Args:
            data_root: Path to tartanair data (e.g., /path/to/tartanair)
            image_size: Resize images to this size (H, W)
            sequence_length: Number of consecutive frames (2 for pairs)
            environments: List of environments to use (e.g., ['office', 'neighborhood'])
            trajectories: List of trajectory IDs to use (e.g., ['P000', 'P001'])
        """
        super().__init__()

        self.data_root = Path(data_root)
        self.image_size = image_size
        self.sequence_length = sequence_length

        # Default to all available data
        if environments is None:
            environments = ['abandonedfactory']
        # Default train/val split (80/20)
        if trajectories is None:
            # All available trajectories
            all_trajs = ['P000', 'P001', 'P002', 'P004', 'P005', 'P006', 'P008', 'P009', 'P010', 'P011']
            # Use environment variable or default split
            split = os.environ.get('DATASET_SPLIT', 'train')
            if split == 'train':
                trajectories = ['P000', 'P001', 'P002', 'P004', 'P005', 'P006', 'P008']  # 70% train
            elif split == 'val':
                trajectories = ['P009', 'P010', 'P011']  # 30% validation
            else:
                trajectories = all_trajs  # Use all if split not specified

        # Collect all frame pairs
        self.samples = []
        self._build_dataset(environments, trajectories)

        # Limit dataset size if specified
        if max_samples is not None and len(self.samples) > max_samples:
            import random
            random.seed(42)  # Reproducible
            self.samples = random.sample(self.samples, max_samples)

        print(f"TartanAir Dataset: {len(self.samples)} frame pairs (monocular RGB)")

    def _build_dataset(self, environments, trajectories):
        """Build list of valid frame pairs with poses."""

        for env in environments:
            for traj in trajectories:
                # TartanAir structure: env/Easy/traj/image_left/*.png
                traj_path = self.data_root / env / 'Easy' / traj

                if not traj_path.exists():
                    continue

                # Get RGB images (monocular left camera only)
                image_dir = traj_path / 'image_left'
                pose_file = traj_path / 'pose_left.txt'

                if not image_dir.exists() or not pose_file.exists():
                    continue

                # Load all images
                image_files = sorted(image_dir.glob('*.png'))

                # Load poses
                poses = self._load_poses(pose_file)

                if len(image_files) != len(poses):
                    print(f"Warning: {env}/{traj} image/pose count mismatch")
                    continue

                # Create consecutive pairs
                for i in range(len(image_files) - 1):
                    self.samples.append({
                        'image0': image_files[i],
                        'image1': image_files[i + 1],
                        'pose0': poses[i],
                        'pose1': poses[i + 1],
                        'env': env,
                        'traj': traj,
                    })

    def _load_poses(self, pose_file):
        """
        Load TartanAir poses from text file.

        Format: Each line is "x y z qx qy qz qw"
        Returns: List of 4x4 transformation matrices
        """
        poses = []

        with open(pose_file, 'r') as f:
            for line in f:
                values = list(map(float, line.strip().split()))

                # Parse position and quaternion
                x, y, z = values[0], values[1], values[2]
                qx, qy, qz, qw = values[3], values[4], values[5], values[6]

                # Convert to 4x4 matrix
                T = self._quaternion_to_matrix(x, y, z, qx, qy, qz, qw)
                poses.append(T)

        return poses

    def _quaternion_to_matrix(self, x, y, z, qx, qy, qz, qw):
        """Convert position + quaternion to 4x4 transformation matrix."""

        # Rotation matrix from quaternion
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ])

        # 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]

        return T

    def _compute_relative_pose(self, T0, T1):
        """
        Compute relative transformation from frame 0 to frame 1.

        Returns:
            R: 3x3 rotation matrix
            t: 3x1 translation vector
        """
        # Relative transformation: T_rel = T0^-1 @ T1
        T_rel = np.linalg.inv(T0) @ T1

        R = T_rel[:3, :3]
        t = T_rel[:3, 3]

        return R, t

    def _load_image(self, path):
        """Load and preprocess monocular RGB image."""

        # Load RGB image
        img = Image.open(path).convert('RGB')

        # Resize
        img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)

        # Convert to tensor and normalize
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a frame pair with relative pose.

        Returns:
            dict with:
                'image0': (3, H, W) RGB image at time t
                'image1': (3, H, W) RGB image at time t+1
                'R_gt': (3, 3) relative rotation
                't_gt': (3,) relative translation
                'K': (3, 3) camera intrinsics (TartanAir default)
        """
        sample = self.samples[idx]

        # Load monocular RGB images
        image0 = self._load_image(sample['image0'])
        image1 = self._load_image(sample['image1'])

        # Compute relative pose
        R_gt, t_gt = self._compute_relative_pose(sample['pose0'], sample['pose1'])

        # TartanAir camera intrinsics (640x480 -> rescaled to our image_size)
        # Original: fx=320, fy=320, cx=320, cy=240
        scale_x = self.image_size[1] / 640.0
        scale_y = self.image_size[0] / 480.0

        K = np.array([
            [320.0 * scale_x, 0, 320.0 * scale_x],
            [0, 320.0 * scale_y, 240.0 * scale_y],
            [0, 0, 1]
        ], dtype=np.float32)

        return {
            'image0': image0,
            'image1': image1,
            'R_gt': torch.from_numpy(R_gt.astype(np.float32)),
            't_gt': torch.from_numpy(t_gt.astype(np.float32)),
            'K': torch.from_numpy(K),
        }


def create_dataloader(
    data_root,
    batch_size=4,
    num_workers=4,
    shuffle=True,
    image_size=(224, 224),
    environments=None,
    trajectories=None,
    max_samples=None,
):
    """
    Create TartanAir dataloader for monocular RGB VO.

    Args:
        data_root: Path to tartanair data
        batch_size: Batch size
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle data
        image_size: Resize images to this size
        environments: List of environments to use
        trajectories: List of trajectory IDs to use

    Returns:
        DataLoader instance
    """
    dataset = TartanAirVO(
        data_root=data_root,
        image_size=image_size,
        environments=environments,
        trajectories=trajectories,
        max_samples=max_samples,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader


def test_dataloader():
    """Test the dataloader (once data is downloaded)."""
    print("\n" + "=" * 70)
    print("Testing TartanAir Monocular RGB Dataloader")
    print("=" * 70)

    data_root = Path(__file__).parent.parent / 'data' / 'tartanair'

    if not data_root.exists() or not any(data_root.iterdir()):
        print(f"\n✗ Data not found at {data_root}")
        print("Please run download_tartanair.py first")
        return

    # Create dataloader
    loader = create_dataloader(
        data_root=data_root,
        batch_size=2,
        num_workers=0,  # Use 0 for testing
        shuffle=False,
        image_size=(224, 224),
    )

    print(f"\n✓ Dataset size: {len(loader.dataset)} frame pairs")
    print(f"✓ Batch size: {loader.batch_size}")
    print(f"✓ Number of batches: {len(loader)}")

    # Test a batch
    print("\nTesting batch loading...")
    batch = next(iter(loader))

    print(f"\n✓ Batch contents:")
    print(f"  image0: {batch['image0'].shape} (monocular RGB)")
    print(f"  image1: {batch['image1'].shape} (monocular RGB)")
    print(f"  R_gt: {batch['R_gt'].shape}")
    print(f"  t_gt: {batch['t_gt'].shape}")
    print(f"  K: {batch['K'].shape}")

    # Verify data ranges
    print(f"\n✓ Data ranges:")
    print(f"  image0: [{batch['image0'].min():.3f}, {batch['image0'].max():.3f}]")
    print(f"  image1: [{batch['image1'].min():.3f}, {batch['image1'].max():.3f}]")
    print(f"  t_gt norm: {torch.norm(batch['t_gt'], dim=1).mean():.3f} (avg)")

    # Check rotation matrices are valid
    R = batch['R_gt'][0]
    det = torch.det(R)
    print(f"  R_gt det(R): {det:.6f} (should be ~1.0)")

    print("\n" + "=" * 70)
    print("✓ Dataloader test complete!")
    print("=" * 70)


if __name__ == '__main__':
    test_dataloader()
