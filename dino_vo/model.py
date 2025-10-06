#!/usr/bin/env python3
"""
DINO-VO Model: DINOv3 + GNN Matcher + Pose Solver

Integrates:
1. DINOv3 feature extraction (frozen)
2. Barbara's GNN matcher (adapted for 384-dim)
3. Weighted 8-point + bundle adjustment
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add e2e_multi_view_matching to path
E2E_PATH = Path(__file__).parent.parent / 'e2e_multi_view_matching-master'
sys.path.insert(0, str(E2E_PATH))

from feature_extractor import DINOv3Extractor
from models.models.multi_view_matcher import MultiViewMatcher


class DINOVO(nn.Module):
    """
    End-to-end monocular RGB visual odometry with DINOv3 features.

    Architecture:
    1. DINOv3 ViT-S/16 (frozen) → 384-dim descriptors
    2. Projection layer: 384 → 256 dim
    3. GNN matcher (trainable)
    4. Pose solver (differentiable)
    """

    def __init__(
        self,
        num_keypoints=256,
        image_size=224,
        dinov3_weights=None,
        matcher_config=None,
    ):
        """
        Args:
            num_keypoints: Number of keypoints to extract
            image_size: Input image size
            dinov3_weights: Path to DINOv3 weights
            matcher_config: Config for GNN matcher
        """
        super().__init__()

        # 1. DINOv3 Feature Extractor (frozen)
        self.feature_extractor = DINOv3Extractor(
            num_keypoints=num_keypoints,
            image_size=image_size,
            weights_path=dinov3_weights,
        )

        # 2. Projection layer: 384 → 256 to match matcher expected dim
        self.descriptor_proj = nn.Conv1d(384, 256, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.descriptor_proj.weight)
        nn.init.constant_(self.descriptor_proj.bias, 0.0)

        # 3. GNN Matcher
        if matcher_config is None:
            matcher_config = {
                'descriptor_dim': 256,
                'weights': 'none',  # Train from scratch
                'keypoint_encoder': [32, 64, 128, 256],
                'GNN_layers': ['self', 'cross'] * 9,
                'sinkhorn_iterations': 100,
                'multi_frame_matching': False,  # Two-view for monocular VO
                'full_output': True,
                'conf_mlp': True,
            }

        self.matcher = MultiViewMatcher(matcher_config)

        print("✓ DINO-VO Model initialized:")
        print(f"  - DINOv3 ViT-S/16 (frozen): 384-dim features")
        print(f"  - Projection: 384 → 256 dim")
        print(f"  - GNN Matcher: {len(matcher_config['GNN_layers'])} layers")
        print(f"  - Keypoints: {num_keypoints} per image")

    def forward(self, batch):
        """
        Forward pass for monocular RGB frame pairs.

        Args:
            batch: Dict with 'image0', 'image1', 'K', 'R_gt', 't_gt'

        Returns:
            Dict with matches, scores, and optionally pose estimates
        """
        image0 = batch['image0']  # (B, 3, H, W)
        image1 = batch['image1']  # (B, 3, H, W)

        # Extract DINOv3 features (frozen)
        with torch.no_grad():
            feat0 = self.feature_extractor(image0)
            feat1 = self.feature_extractor(image1)

        # Project descriptors: 384 → 256
        desc0 = self.descriptor_proj(feat0['descriptors'])  # (B, 256, N)
        desc1 = self.descriptor_proj(feat1['descriptors'])  # (B, 256, N)

        # Prepare data for matcher
        matcher_data = {
            'image0': image0,
            'image1': image1,
            'keypoints0': feat0['keypoints'],
            'keypoints1': feat1['keypoints'],
            'descriptors0': desc0,
            'descriptors1': desc1,
            'scores0': feat0['scores'],
            'scores1': feat1['scores'],
        }

        # Run GNN matcher
        result = self.matcher.match(matcher_data, id0=0, id1=1)

        # Add original features to result
        result.update({
            'keypoints0': feat0['keypoints'],
            'keypoints1': feat1['keypoints'],
            'descriptors0': desc0,
            'descriptors1': desc1,
            'scores0': feat0['scores'],
            'scores1': feat1['scores'],
        })

        return result

    def trainable_parameters(self):
        """Get trainable parameters (exclude frozen DINOv3)."""
        params = []
        params += list(self.descriptor_proj.parameters())
        params += list(self.matcher.parameters())
        return params


def test_model():
    """Test the integrated model."""
    print("\n" + "=" * 70)
    print("Testing DINO-VO Model (DINOv3 + GNN Matcher)")
    print("=" * 70)

    # Create model
    model = DINOVO(
        num_keypoints=256,
        image_size=224,
    ).cuda().eval()

    # Test batch
    batch = {
        'image0': torch.randn(2, 3, 224, 224).cuda(),
        'image1': torch.randn(2, 3, 224, 224).cuda(),
        'K': torch.eye(3).unsqueeze(0).repeat(2, 1, 1).cuda(),
    }

    print("\nRunning forward pass...")
    with torch.no_grad():
        result = model(batch)

    print(f"\n✓ Model outputs:")
    for key in result.keys():
        if isinstance(result[key], torch.Tensor):
            print(f"  {key}: {result[key].shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.trainable_parameters())
    frozen_params = total_params - trainable_params

    print(f"\n✓ Parameter count:")
    print(f"  Total: {total_params / 1e6:.2f}M")
    print(f"  Trainable: {trainable_params / 1e6:.2f}M")
    print(f"  Frozen (DINOv3): {frozen_params / 1e6:.2f}M")

    print(f"\n✓ VRAM usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print("=" * 70)


if __name__ == '__main__':
    test_model()
