"""Model definitions. All models take 1 x H x W grayscale input and return 2 logits."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


class SmallCNN(nn.Module):
    """Transparent baseline: 3 conv blocks + global average pooling."""

    def __init__(self, in_channels: int = 1, num_classes: int = 2, width: int = 32, dropout: float = 0.2):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            )

        self.block1 = block(in_channels, width)
        self.block2 = block(width, 2 * width)
        self.block3 = block(2 * width, 4 * width)
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(4 * width, num_classes))

    def features(self, x):
        """Last conv feature map (B x 4*width x H/4 x W/4), the Grad-CAM layer's output."""
        return self.block3(self.pool(self.block2(self.pool(self.block1(x)))))

    def forward(self, x):
        return self.head(self.features(x))

    def cam(self, feats, cls):
        """Class activation maps (Zhou et al., CVPR 2016) for class indices ``cls`` (B,).

        With global average pooling + one linear layer, logit = mean(CAM) + bias, so this is
        exact and differentiable (Grad-CAM gives the same map up to a positive scale).
        """
        w = self.head[-1].weight[cls]  # B x K
        return torch.einsum("bkhw,bk->bhw", feats, w)

    @property
    def gradcam_layer(self) -> nn.Module:
        return self.block3


class ResNet18Small(nn.Module):
    """torchvision ResNet-18 trained from scratch, adapted to small grayscale inputs.

    The 7x7/stride-2 stem and max-pool are replaced by a 3x3/stride-1 stem (as is
    common for 28-64 px inputs, cf. the MedMNIST v2 benchmark's ResNet-18 at 28 px).
    No ImageNet weights are used, so no third-party weight licence applies.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()
        net = resnet18(weights=None, num_classes=num_classes)
        net.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False)
        net.maxpool = nn.Identity()
        self.net = net

    def forward(self, x):
        return self.net(x)

    @property
    def gradcam_layer(self) -> nn.Module:
        return self.net.layer4


MODELS = {"small_cnn": SmallCNN, "resnet18": ResNet18Small}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"unknown model '{name}'; choose from {sorted(MODELS)}")
    return MODELS[name](**kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path: str, model: nn.Module, meta: dict) -> None:
    """Save weights as a plain state_dict plus JSON-serialisable metadata."""
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> tuple[nn.Module, dict]:
    """Load a checkpoint written by :func:`save_checkpoint`.

    ``weights_only=True`` refuses arbitrary pickled code, so only tensors and
    plain containers are deserialised (TESTING.md: untrusted serialized files).
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt or "meta" not in ckpt:
        raise ValueError(f"{path} is not a medxai checkpoint")
    meta = ckpt["meta"]
    model = build_model(meta["model"], **meta.get("model_kwargs", {}))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, meta
