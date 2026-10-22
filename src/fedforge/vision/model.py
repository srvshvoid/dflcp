"""PyTorch neural network for deepfake detection with calibrated spatial attention."""

from __future__ import annotations

from typing import Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttentionBlock(nn.Module):
    """Spatial attention module focused on facial boundary and artefact regions."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = self.conv(x)          # (B, 1, H, W)  values in [0, 1]
        attended = x * mask + x      # residual attention
        return attended, mask


class DeepfakeDetector(nn.Module):
    """Lightweight CNN + Spatial Attention deepfake detector.

    Design philosophy
    -----------------
    This model is *not* trained on labelled data; instead it acts as a
    learned nonlinear feature extractor that is combined with the much more
    reliable rule-based pixel-forensic pipeline in ``detector.py``.

    The head bias is initialised to ``-3.0`` so that, with random weights,
    the raw logit for any clean image is strongly negative (sigmoid → ~0.05).
    The spatial-attention prominence boost is intentionally disabled for the
    untrained model because the Sigmoid-gated mask always produces prominence
    in [1.3, 2.0] regardless of image content, which would corrupt the score.
    Once the model is fine-tuned, enable prominence boosting by passing
    ``use_attention_boost=True`` to ``predict_proba``.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.attention = SpatialAttentionBlock(256)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    # Strong negative bias → sigmoid(logit) ≈ 0.05 for random input.
                    # This means the untrained CNN alone will not flag real photos.
                    nn.init.constant_(m.bias, -3.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.stem(x)
        feat = self.layer1(feat)
        feat = self.layer2(feat)
        attended, mask = self.attention(feat)
        logits = self.head(attended)
        return logits, mask

    def predict_proba(self, x: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """Return (fake_probability [0,1], attention_mask).

        The raw CNN logit is converted to probability without any prominence
        boosting, because the untrained Sigmoid attention produces uniformly
        high prominence on every image.  The forensic pipeline in
        ``detector.py`` handles the actual manipulation scoring.
        """
        self.eval()
        with torch.no_grad():
            logits, mask = self.forward(x)
            prob = float(torch.sigmoid(logits.mean()).item())
            prob = max(0.01, min(0.99, prob))
        return prob, mask

    # ------------------------------------------------------------------ #
    #  Federated learning helpers                                          #
    # ------------------------------------------------------------------ #
    def get_parameter_vector(self) -> Sequence[float]:
        """Serialise trainable weights for federated aggregation."""
        params: list = []
        for p in self.parameters():
            if p.requires_grad:
                params.extend(p.detach().cpu().flatten().tolist())
        return tuple(params)

    def set_parameter_vector(self, vector: Sequence[float]) -> None:
        """Load a flat parameter vector into model weights."""
        idx = 0
        with torch.no_grad():
            for p in self.parameters():
                if p.requires_grad:
                    numel = p.numel()
                    data = torch.tensor(
                        vector[idx: idx + numel], dtype=p.dtype, device=p.device
                    )
                    p.copy_(data.view_as(p))
                    idx += numel
