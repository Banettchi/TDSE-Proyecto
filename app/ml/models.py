"""
Deep Learning model definitions for skin lesion classification.
Three architectures as described in the technical document:
  - EfficientNet-B7: Highest AUC, higher latency (server, high precision)
  - ResNet-50: Balanced throughput/accuracy (server, baseline)
  - ViT-Base: Best cross-domain robustness (transformer-based)

All models use ImageNet pretrained backbones with custom classification
heads for 7-class HAM10000 classification.
"""
import os
import torch
import torch.nn as nn
import torchvision.models as tv_models
from app.config import settings

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class EfficientNetB7Classifier(nn.Module):
    """
    EfficientNet-B7 for high-precision skin lesion classification.
    AUC: 0.88–0.92 | Latency: 120–200ms (GPU T4)
    Best for: Maximum diagnostic sensitivity.
    """

    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super().__init__()
        weights = tv_models.EfficientNet_B7_Weights.DEFAULT if pretrained else None
        self.backbone = tv_models.efficientnet_b7(weights=weights)

        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def get_feature_layer(self):
        """Returns the last convolutional layer for Grad-CAM."""
        return self.backbone.features[-1]


class ResNet50Classifier(nn.Module):
    """
    ResNet-50 for high-throughput skin lesion classification.
    AUC: 0.83–0.87 | Latency: 45–80ms (GPU T4)
    Best for: High volume screening with moderate accuracy.
    """

    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super().__init__()
        weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = tv_models.resnet50(weights=weights)

        # Replace final FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def get_feature_layer(self):
        """Returns layer4 for Grad-CAM."""
        return self.backbone.layer4[-1]


class ViTBaseClassifier(nn.Module):
    """
    Vision Transformer (ViT-Base) for cross-domain robust classification.
    AUC: 0.867 | Latency: ~100ms (GPU T4)
    Best for: Robustness under variable capture conditions.
    Uses timm library if available, otherwise falls back to torchvision.
    """

    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super().__init__()

        if TIMM_AVAILABLE:
            self.backbone = timm.create_model(
                'vit_base_patch16_224',
                pretrained=pretrained,
                num_classes=0  # Remove head
            )
            in_features = self.backbone.num_features
        else:
            weights = tv_models.ViT_B_16_Weights.DEFAULT if pretrained else None
            self.backbone = tv_models.vit_b_16(weights=weights)
            in_features = self.backbone.heads.head.in_features
            self.backbone.heads = nn.Identity()

        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def get_feature_layer(self):
        """Returns the last transformer block for attention visualization."""
        if TIMM_AVAILABLE:
            return self.backbone.blocks[-1].norm1
        else:
            return self.backbone.encoder.layers[-1].ln_1


# ── Model Registry ─────────────────────────────────────────────
MODEL_REGISTRY = {
    "efficientnet": {
        "class": EfficientNetB7Classifier,
        "input_size": settings.IMAGE_SIZE_EFFICIENTNET,
        "display_name": "EfficientNet-B7",
        "context": "Servidor, alta precisión",
        "auc_range": "0.88–0.92",
        "latency_range": "120–200ms"
    },
    "resnet": {
        "class": ResNet50Classifier,
        "input_size": settings.IMAGE_SIZE_RESNET,
        "display_name": "ResNet-50",
        "context": "Servidor, base",
        "auc_range": "0.83–0.87",
        "latency_range": "45–80ms"
    },
    "vit": {
        "class": ViTBaseClassifier,
        "input_size": settings.IMAGE_SIZE_VIT,
        "display_name": "ViT-Base",
        "context": "Robustez cross-domain",
        "auc_range": "0.86–0.89",
        "latency_range": "80–120ms"
    }
}


def load_model(model_name: str, device: str = "cpu") -> tuple:
    """
    Load a model by name. Attempts to load fine-tuned weights first,
    falls back to pretrained ImageNet backbone.

    Returns:
        (model, input_size)
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    info = MODEL_REGISTRY[model_name]
    model_class = info["class"]

    # Try to load fine-tuned weights
    weight_path = os.path.join(settings.MODEL_DIR, f"{model_name}_skin_lesion.pth")

    if os.path.exists(weight_path):
        model = model_class(num_classes=settings.NUM_CLASSES, pretrained=False)
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"  OK - Loaded fine-tuned weights for {info['display_name']}")
    else:
        model = model_class(num_classes=settings.NUM_CLASSES, pretrained=True)
        print(f"  WARN - Using ImageNet pretrained backbone for {info['display_name']} (no fine-tuned weights)")

    model = model.to(device)
    model.eval()
    return model, info["input_size"]
