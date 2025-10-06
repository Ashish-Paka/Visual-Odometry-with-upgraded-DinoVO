#!/usr/bin/env python3
"""
DINOv3 Feature Extractor for Monocular RGB Visual Odometry

Replaces SuperPoint with DINOv3 for feature extraction.
- Input: Monocular RGB images
- Output: Keypoints, descriptors, scores (same format as SuperPoint)
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add dinov3 to path
DINOV3_PATH = Path(__file__).parent.parent / 'dinov3'
sys.path.insert(0, str(DINOV3_PATH))


class DINOv3Extractor(nn.Module):
    """
    Extract keypoints and descriptors from monocular RGB using DINOv3.

    This module:
    1. Loads frozen DINOv3 ViT-S/16 backbone
    2. Extracts dense patch features
    3. Detects salient keypoints via feature magnitude
    4. Returns keypoints + descriptors in SuperPoint-compatible format
    """

    def __init__(
        self,
        num_keypoints=256,
        image_size=224,
        model_name='dinov3_vits16',
        weights_path=None,
    ):
        """
        Args:
            num_keypoints: Number of keypoints to extract per image
            image_size: Input image size (square)
            model_name: DINOv3 model variant
            weights_path: Path to pretrained weights
        """
        super().__init__()

        self.num_keypoints = num_keypoints
        self.image_size = image_size
        self.patch_size = 16  # DINOv3 ViT-S/16 uses 16x16 patches
        self.feat_dim = 384   # DINOv3-small feature dimension

        # Load DINOv3 model (frozen)
        print(f"Loading DINOv3 {model_name}...")
        if weights_path is None:
            weights_path = Path(__file__).parent.parent / 'dinov3_weights' / 'dinov3_vits16_pretrain_lvd1689m.pth'

        repo_path = str(DINOV3_PATH)
        self.dino = torch.hub.load(
            repo_path,
            model_name,
            source='local',
            weights=str(weights_path)
        )

        # Freeze DINOv3 (we only train the matcher)
        for param in self.dino.parameters():
            param.requires_grad = False

        self.dino.eval()

        print(f"✓ DINOv3 loaded: {num_keypoints} keypoints, {image_size}px images")

    @torch.no_grad()
    def forward(self, images):
        """
        Extract features from monocular RGB images.

        Args:
            images: (B, 3, H, W) RGB images

        Returns:
            dict with:
                'keypoints': (B, N, 2) - pixel coordinates [x, y]
                'descriptors': (B, D, N) - feature descriptors (D=384)
                'scores': (B, N) - keypoint confidence scores
        """
        B, C, H, W = images.shape
        assert C == 3, f"Expected RGB images (3 channels), got {C}"

        # Resize if needed
        if H != self.image_size or W != self.image_size:
            images = torch.nn.functional.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            )

        # Extract DINOv3 features
        features = self.dino(images)  # (B, feat_dim)

        # Note: DINOv3 output is CLS token only by default
        # We need to get patch tokens for dense features
        # Let's use forward_features to get all tokens
        with torch.no_grad():
            # Get all tokens (CLS + patches)
            all_tokens = self.dino.forward_features(images)['x_norm_patchtokens']
            # all_tokens shape: (B, num_patches, feat_dim)

        # Detect keypoints via feature saliency
        keypoints, descriptors, scores = self._detect_keypoints(
            all_tokens,
            self.num_keypoints,
            H, W
        )

        return {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'scores': scores,
        }

    def _detect_keypoints(self, patch_features, num_kpts, orig_h, orig_w):
        """
        Detect salient keypoints from patch features.

        Args:
            patch_features: (B, num_patches, feat_dim)
            num_kpts: number of keypoints to select
            orig_h, orig_w: original image size

        Returns:
            keypoints: (B, N, 2) - [x, y] in original image coordinates
            descriptors: (B, feat_dim, N)
            scores: (B, N)
        """
        B, num_patches, feat_dim = patch_features.shape

        # Compute saliency as L2 norm of features
        saliency = torch.norm(patch_features, dim=2)  # (B, num_patches)

        # Always select exactly num_kpts keypoints
        # If we have fewer patches than requested, we need to pad
        actual_num_kpts = min(num_kpts, num_patches)
        scores, indices = torch.topk(saliency, k=actual_num_kpts, dim=1)  # (B, actual_N)

        # Pad to exact num_kpts if needed
        if actual_num_kpts < num_kpts:
            pad_size = num_kpts - actual_num_kpts
            scores = torch.cat([scores, torch.zeros(B, pad_size, device=scores.device)], dim=1)
            # Repeat last index for padding (will give duplicate keypoints but maintains shape)
            indices = torch.cat([indices, indices[:, -1:].repeat(1, pad_size)], dim=1)

        # Convert patch indices to pixel coordinates
        grid_size = int(num_patches ** 0.5)  # Assuming square grid
        scale_x = orig_w / self.image_size
        scale_y = orig_h / self.image_size

        keypoints = []
        for b in range(B):
            kpts = []
            for idx in indices[b]:
                patch_idx = idx.item()

                # Get patch grid position
                row = patch_idx // grid_size
                col = patch_idx % grid_size

                # Convert to pixel coordinates (center of patch)
                x = (col + 0.5) * self.patch_size * scale_x
                y = (row + 0.5) * self.patch_size * scale_y

                kpts.append([x, y])

            keypoints.append(kpts)

        keypoints = torch.tensor(
            keypoints,
            dtype=torch.float32,
            device=patch_features.device
        )  # (B, N, 2)

        # Gather descriptors for selected keypoints
        # indices shape: (B, N)
        # We need to index patch_features: (B, num_patches, feat_dim)
        descriptors = torch.gather(
            patch_features,
            1,
            indices.unsqueeze(-1).expand(-1, -1, feat_dim)
        )  # (B, N, feat_dim)

        # Transpose to match SuperPoint format: (B, feat_dim, N)
        descriptors = descriptors.transpose(1, 2)

        return keypoints, descriptors, scores


def test_extractor():
    """Test the feature extractor"""
    print("\n" + "=" * 60)
    print("Testing DINOv3 Feature Extractor for Monocular RGB")
    print("=" * 60)

    # Create extractor
    extractor = DINOv3Extractor(
        num_keypoints=256,
        image_size=224
    ).cuda().eval()

    # Test with monocular RGB images
    print("\nTesting with batch of monocular RGB images...")
    batch_rgb = torch.randn(2, 3, 480, 640).cuda()  # 2 images, RGB, different size

    with torch.no_grad():
        output = extractor(batch_rgb)

    print(f"\n✓ Input: {batch_rgb.shape} (monocular RGB)")
    print(f"✓ Keypoints: {output['keypoints'].shape}")
    print(f"✓ Descriptors: {output['descriptors'].shape}")
    print(f"✓ Scores: {output['scores'].shape}")
    print(f"✓ VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Check format compatibility with SuperPoint
    # Note: DINOv3 ViT-S/16 with 224x224 produces 14x14 = 196 patches
    assert output['keypoints'].shape[0] == 2, "Batch size mismatch"
    assert output['keypoints'].shape[2] == 2, "Keypoint coordinates should be (x, y)"
    assert output['descriptors'].shape[0] == 2, "Batch size mismatch"
    assert output['descriptors'].shape[1] == 384, "Feature dimension should be 384"
    assert output['scores'].shape[0] == 2, "Batch size mismatch"

    print(f"\n✓ Output format compatible with SuperPoint!")
    print(f"  (Using all {output['keypoints'].shape[1]} patches as keypoints)")
    print("=" * 60)


if __name__ == '__main__':
    test_extractor()
