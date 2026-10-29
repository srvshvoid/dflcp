"""Neural network architectures and federated state serialization for FedForge.

Architecture choices:
1. ResNet18 (torchvision) with 1-logit output head for binary synthetic detection.
2. ImageNet-pretrained backbone with option to freeze backbone on CPU.
3. Fallback SmallCNN module if pretrained weights are unavailable or offline.
4. Comprehensive state serialization (parameters + BatchNorm running statistics).
5. Flattening and unflattening helpers for federated aggregation.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


# --------------------------------------------------------------------------- #
# Lightweight Fallback Architecture (SmallCNN)                               #
# --------------------------------------------------------------------------- #
class SmallCNN(nn.Module):
    """Compact 4-layer CNN for binary deepfake detection on low-resource hardware."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = torch.flatten(feat, 1)
        return self.fc(feat)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> Tuple[float, Optional[torch.Tensor]]:
        """Return sigmoid probability and dummy mask for pipeline compatibility."""
        self.eval()
        logits = self.forward(x)
        prob = float(torch.sigmoid(logits).squeeze().item())
        return prob, None


# --------------------------------------------------------------------------- #
# ResNet18 Binary Classifier                                                  #
# --------------------------------------------------------------------------- #
class ResNet18Binary(nn.Module):
    """ResNet18 backbone with a 1-logit linear classification head.

    Head outputs a single raw logit z:
        P(fake) = sigmoid(z)
        Loss = BCEWithLogitsLoss(z, target)
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # Replace 1000-class ImageNet head with 1-logit binary head
        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Linear(in_features, 1)

        # Initialise linear head weights with Xavier uniform, zero bias
        nn.init.xavier_uniform_(self.backbone.fc.weight)
        nn.init.constant_(self.backbone.fc.bias, 0.0)

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self._set_backbone_trainable(False)

    def _set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze backbone layers, keeping only fc trainable."""
        for name, param in self.backbone.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> Tuple[float, Optional[torch.Tensor]]:
        """Return sigmoid probability and dummy mask for pipeline compatibility."""
        self.eval()
        logits = self.forward(x)
        prob = float(torch.sigmoid(logits).squeeze().item())
        return prob, None


# --------------------------------------------------------------------------- #
# Factory Builder                                                             #
# --------------------------------------------------------------------------- #
def build_model(
    model_type: str = "resnet18",
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Build and configure the classification model.

    Args:
        model_type: 'resnet18' (standard) or 'smallcnn' (fallback).
        pretrained: Load ImageNet pretrained weights if available.
        freeze_backbone: If True, only the linear head (513 params) is trained.
                         Recommended on CPU for fast federated convergence.

    Returns:
        Configured PyTorch nn.Module.
    """
    if model_type.lower() == "resnet18":
        try:
            return ResNet18Binary(pretrained=pretrained, freeze_backbone=freeze_backbone)
        except Exception:
            # Fallback to non-pretrained ResNet18 if network download fails
            return ResNet18Binary(pretrained=False, freeze_backbone=freeze_backbone)
    elif model_type.lower() == "smallcnn":
        return SmallCNN()
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'resnet18' or 'smallcnn'.")


# --------------------------------------------------------------------------- #
# Federated State Utilities (Parameters + BatchNorm Buffers)                  #
# --------------------------------------------------------------------------- #
def get_state(
    model: nn.Module,
    scope: str = "head",
) -> Dict[str, torch.Tensor]:
    """Extract model state covering parameters and BatchNorm buffers.

    Args:
        model: PyTorch module.
        scope:
            'head' -> Only classification head parameters ('fc.weight', 'fc.bias' or 'backbone.fc.*').
                      Compact vector: 513 float32 values (2,052 bytes).
            'full' -> Full model state dict, including all layer weights, biases,
                      and BatchNorm buffers (running_mean, running_var, num_batches_tracked).

    Returns:
        Ordered dictionary of detached tensors on CPU.
    """
    full_dict = model.state_dict()
    if scope == "head":
        # Extract only the final classification linear layer
        head_dict: Dict[str, torch.Tensor] = OrderedDict()
        for k, v in full_dict.items():
            if "fc." in k:
                head_dict[k] = v.detach().clone().cpu()
        return head_dict
    elif scope == "full":
        return OrderedDict({k: v.detach().clone().cpu() for k, v in full_dict.items()})
    else:
        raise ValueError(f"Unknown scope '{scope}'. Allowed: 'head', 'full'.")


def set_state(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
    strict: bool = False,
) -> None:
    """Load parameter and buffer tensors into model with strict/non-strict verification."""
    # Ensure all state tensors are loaded to the target model's device
    device = next(model.parameters()).device
    converted_state: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            converted_state[k] = v.to(device)
        else:
            converted_state[k] = torch.tensor(v, device=device)

    model.load_state_dict(converted_state, strict=strict)


# --------------------------------------------------------------------------- #
# Flatten / Unflatten Serialization for Federated Vectors                     #
# --------------------------------------------------------------------------- #
def get_state_schema(
    state: Mapping[str, torch.Tensor]
) -> Dict[str, Tuple[torch.Size, torch.dtype]]:
    """Record tensor shapes and dtypes for exact reconstruction."""
    return {k: (v.shape, v.dtype) for k, v in state.items()}


def state_to_vector(state: Mapping[str, torch.Tensor]) -> Tuple[float, ...]:
    """Serialize an ordered state dict into a flat float tuple for federated aggregation."""
    flat_values: List[float] = []
    # Sort keys for deterministic parameter alignment across clients
    for k in sorted(state.keys()):
        tensor = state[k].detach().cpu().flatten()
        flat_values.extend(tensor.float().tolist())
    return tuple(flat_values)


def vector_to_state(
    flat_vector: Sequence[float],
    schema: Mapping[str, Tuple[torch.Size, torch.dtype]],
) -> Dict[str, torch.Tensor]:
    """Reconstruct an ordered state dict from a flat parameter vector and schema."""
    state: Dict[str, torch.Tensor] = OrderedDict()
    idx = 0
    for k in sorted(schema.keys()):
        shape, dtype = schema[k]
        numel = shape.numel()
        chunk = flat_vector[idx: idx + numel]
        if len(chunk) != numel:
            raise ValueError(
                f"Flat vector length mismatch for key '{k}'. Expected {numel} elements, got {len(chunk)}"
            )
        tensor = torch.tensor(chunk, dtype=dtype).view(shape)
        state[k] = tensor
        idx += numel

    if idx != len(flat_vector):
        raise ValueError(
            f"Extra values in flat vector: consumed {idx} of {len(flat_vector)} elements."
        )
    return state


# --------------------------------------------------------------------------- #
# Architectural Decision Summary (Federated Transmission Contract)            #
# --------------------------------------------------------------------------- #
FEDERATION_CONTRACT = {
    "default_scope": "head",
    "head_parameter_count": 513,
    "head_byte_size": 2052,  # 513 * 4 bytes (float32)
    "full_parameter_count": 11177025,
    "full_byte_size": 44708100,  # ~44.7 MB (float32)
    "rationale": (
        "Head-only federation transmits 513 parameters (2.05 KB/client/round) rather than "
        "44.7 MB. It enables sub-second federated aggregation, operates directly with "
        "robust.py pure-Python aggregators (Krum, Trimmed Mean, Median), and avoids "
        "BatchNorm divergence on non-IID shards when the feature backbone is pretrained."
    ),
}
