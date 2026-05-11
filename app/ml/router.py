"""
Model Router — Dynamic model selection based on tenant config,
image quality, and clinical context.

The router decides which of the three architectures offers the best
precision-latency tradeoff for each case, as described in §3.3.
"""
from typing import Optional
from app.config import settings


class ModelRouter:
    """
    Selects the optimal model based on:
      - Tenant configuration (default model preference)
      - Image source (dermatoscope vs smartphone)
      - Image quality score
      - Explicit user override
    """

    # Decision matrix
    RULES = {
        # (image_source, quality_range) -> recommended model
        ("dermatoscope", "high"): "efficientnet",    # Best AUC with clean images
        ("dermatoscope", "medium"): "efficientnet",
        ("dermatoscope", "low"): "resnet",           # Faster, more tolerant
        ("smartphone", "high"): "vit",               # Best cross-domain robustness
        ("smartphone", "medium"): "vit",
        ("smartphone", "low"): "resnet",             # Most tolerant to noise
    }

    @staticmethod
    def classify_quality(quality_score: float) -> str:
        """Classify quality score into tiers."""
        if quality_score >= 0.7:
            return "high"
        elif quality_score >= 0.4:
            return "medium"
        else:
            return "low"

    @classmethod
    def select_model(cls,
                     tenant_default: str = "efficientnet",
                     image_source: str = "smartphone",
                     quality_score: float = 0.5,
                     user_override: Optional[str] = None,
                     fitzpatrick_type: Optional[int] = None) -> dict:
        """
        Select the optimal model for a given inference request.

        Args:
            tenant_default: Tenant's configured default model
            image_source: 'dermatoscope' or 'smartphone'
            quality_score: Image quality score (0-1)
            user_override: Explicit model selection by user
            fitzpatrick_type: Patient's Fitzpatrick skin type (1-6)

        Returns:
            dict with 'model', 'reason', and 'fallback'
        """
        # User override takes precedence
        if user_override and user_override in ["efficientnet", "resnet", "vit"]:
            return {
                "model": user_override,
                "reason": f"Selección manual del usuario: {user_override}",
                "fallback": tenant_default
            }

        quality_tier = cls.classify_quality(quality_score)

        # For dark skin types (IV-VI), ViT shows more stable performance
        # per the experimental results (§4.1)
        if fitzpatrick_type and fitzpatrick_type >= 4:
            if image_source == "smartphone":
                return {
                    "model": "vit",
                    "reason": (
                        f"ViT seleccionado para fototipo {fitzpatrick_type} "
                        f"(mayor robustez cross-domain, variación AUC máx. 0.048)"
                    ),
                    "fallback": "resnet"
                }

        # Low quality images always go to ResNet (most tolerant)
        if quality_tier == "low":
            return {
                "model": "resnet",
                "reason": (
                    f"ResNet-50 seleccionado por baja calidad de imagen "
                    f"(score: {quality_score:.2f}). Mayor tolerancia a ruido."
                ),
                "fallback": "resnet"
            }

        # Apply decision matrix
        key = (image_source, quality_tier)
        recommended = cls.RULES.get(key, tenant_default)

        # Generate explanation
        reasons = {
            "efficientnet": (
                f"EfficientNet-B7: máxima sensibilidad diagnóstica "
                f"(imagen {image_source}, calidad {quality_tier})"
            ),
            "resnet": (
                f"ResNet-50: alto throughput con precisión moderada "
                f"(imagen {image_source}, calidad {quality_tier})"
            ),
            "vit": (
                f"ViT-Base: robustez cross-domain óptima "
                f"(imagen {image_source}, calidad {quality_tier})"
            )
        }

        return {
            "model": recommended,
            "reason": reasons.get(recommended, "Modelo por defecto del tenant"),
            "fallback": tenant_default
        }

    @staticmethod
    def get_model_info(model_name: str) -> dict:
        """Get model metadata for display."""
        from app.ml.models import MODEL_REGISTRY
        info = MODEL_REGISTRY.get(model_name, {})
        return {
            "name": model_name,
            "display_name": info.get("display_name", model_name),
            "input_size": info.get("input_size", 224),
            "context": info.get("context", ""),
            "auc_range": info.get("auc_range", "N/A"),
            "latency_range": info.get("latency_range", "N/A")
        }
