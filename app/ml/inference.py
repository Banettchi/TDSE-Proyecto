"""
Inference Engine — Orchestrates the full inference pipeline.
Combines preprocessing, model selection, classification, Grad-CAM,
and ABCDE analysis into a single coherent workflow.
"""
import os
import time
import uuid
import torch
import cv2
import numpy as np
from typing import Optional, Dict, Any

from app.config import settings
from app.ml.models import load_model, MODEL_REGISTRY
from app.ml.preprocessing import ImagePreprocessor
from app.ml.gradcam import GradCAM
from app.ml.abcde import ABCDEAnalyzer
from app.ml.router import ModelRouter


class InferenceEngine:
    """
    Central inference engine that manages model lifecycle and
    executes the complete analysis pipeline.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models: Dict[str, tuple] = {}  # {name: (model, input_size)}
        self.preprocessors: Dict[int, ImagePreprocessor] = {}
        self.abcde_analyzer = ABCDEAnalyzer()
        self._loaded = False

    def load_all_models(self):
        """Load all three models into memory."""
        if self._loaded:
            return

        print(f"🔄 Loading models on device: {self.device}")
        for name in MODEL_REGISTRY:
            try:
                model, input_size = load_model(name, self.device)
                self.models[name] = (model, input_size)
                self.preprocessors[input_size] = ImagePreprocessor(target_size=input_size)
                print(f"  ✓ {MODEL_REGISTRY[name]['display_name']} loaded (input: {input_size}x{input_size})")
            except Exception as e:
                print(f"  ✗ Failed to load {name}: {e}")

        self._loaded = True
        print(f"✓ {len(self.models)} models loaded successfully")

    def load_single_model(self, model_name: str):
        """Load a single model on demand."""
        if model_name in self.models:
            return

        model, input_size = load_model(model_name, self.device)
        self.models[model_name] = (model, input_size)
        self.preprocessors[input_size] = ImagePreprocessor(target_size=input_size)

    async def run_inference(
        self,
        image_path: str,
        model_name: Optional[str] = None,
        image_source: str = "smartphone",
        fitzpatrick_type: Optional[int] = None,
        tenant_default_model: str = "efficientnet",
        previous_image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute the full inference pipeline:
          1. Validate image quality
          2. Select optimal model via router
          3. Preprocess image
          4. Run classification
          5. Generate Grad-CAM heatmap
          6. Perform ABCDE analysis
          7. Determine risk level

        Returns comprehensive analysis results.
        """
        start_time = time.time()
        result_id = str(uuid.uuid4())

        # ── Step 1: Load image and validate quality ──
        raw_image = cv2.imread(image_path)
        if raw_image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # Quick quality check
        temp_preprocessor = ImagePreprocessor(target_size=224)
        is_valid, quality_score, quality_msg = temp_preprocessor.validate_quality(raw_image)

        # ── Step 2: Select model via Router ──
        if model_name:
            router_result = ModelRouter.select_model(
                user_override=model_name,
                tenant_default=tenant_default_model
            )
        else:
            router_result = ModelRouter.select_model(
                tenant_default=tenant_default_model,
                image_source=image_source,
                quality_score=quality_score,
                fitzpatrick_type=fitzpatrick_type
            )

        selected_model = router_result["model"]

        # Ensure model is loaded
        if selected_model not in self.models:
            self.load_single_model(selected_model)

        model, input_size = self.models[selected_model]
        preprocessor = self.preprocessors[input_size]

        # ── Step 3: Preprocess ──
        tensor, original_image, _ = preprocessor.preprocess_for_inference(image_path)
        tensor = tensor.to(self.device)

        # ── Step 4: Classification ──
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = probabilities.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()

        # Class probabilities dict
        class_probs = {}
        for i, name in enumerate(settings.CLASS_NAMES):
            class_probs[name] = round(probabilities[0, i].item(), 4)

        # ── Step 5: Grad-CAM ──
        gradcam_path = None
        try:
            target_layer = model.get_feature_layer()
            grad_cam = GradCAM(model, target_layer)

            tensor_grad = tensor.clone().requires_grad_(True)
            heatmap, _, _ = grad_cam.generate(tensor_grad, predicted_class)

            # Save Grad-CAM overlay
            gradcam_filename = f"gradcam_{result_id}.jpg"
            gradcam_path = os.path.join(settings.UPLOAD_DIR, gradcam_filename)
            overlay = grad_cam.overlay_heatmap(original_image, heatmap)
            cv2.imwrite(gradcam_path, overlay)
        except Exception as e:
            print(f"Grad-CAM generation failed: {e}")

        # ── Step 6: ABCDE Analysis ──
        abcde_result = None
        try:
            mask, _ = preprocessor.segment_lesion(original_image)

            # Check for evolution (previous image)
            prev_image = None
            prev_mask = None
            if previous_image_path and os.path.exists(previous_image_path):
                prev_image = cv2.imread(previous_image_path)
                if prev_image is not None:
                    prev_mask, _ = preprocessor.segment_lesion(prev_image)

            abcde_result = self.abcde_analyzer.analyze(
                original_image, mask, prev_image, prev_mask
            )
        except Exception as e:
            print(f"ABCDE analysis failed: {e}")

        # ── Step 7: Risk Level ──
        is_malignant = predicted_class in settings.HIGH_RISK_CLASSES
        risk_level = self._calculate_risk_level(
            predicted_class, confidence, abcde_result
        )

        # Save processed image
        processed_filename = f"processed_{result_id}.jpg"
        processed_path = os.path.join(settings.UPLOAD_DIR, processed_filename)
        processed = preprocessor.normalize_illumination(original_image)
        cv2.imwrite(processed_path, processed)

        elapsed_ms = (time.time() - start_time) * 1000

        return {
            "id": result_id,
            "model_used": selected_model,
            "model_display_name": MODEL_REGISTRY[selected_model]["display_name"],
            "model_selection_reason": router_result["reason"],
            "predicted_class": predicted_class,
            "predicted_class_name": settings.CLASS_NAMES[predicted_class],
            "confidence": round(confidence, 4),
            "is_malignant": is_malignant,
            "risk_level": risk_level,
            "class_probabilities": class_probs,
            "abcde_scores": abcde_result,
            "image_quality_score": round(quality_score, 4),
            "quality_message": quality_msg,
            "original_image_path": image_path,
            "processed_image_path": processed_path,
            "gradcam_image_path": gradcam_path,
            "inference_time_ms": round(elapsed_ms, 2),
            "device": self.device
        }

    def _calculate_risk_level(self, predicted_class: int, confidence: float,
                               abcde_result: Optional[dict]) -> str:
        """
        Determine risk level combining model prediction and ABCDE score.
        """
        # Base risk from prediction
        if predicted_class == settings.MELANOMA_CLASS_INDEX:
            if confidence >= 0.8:
                base_risk = 4  # muy_alto
            elif confidence >= 0.5:
                base_risk = 3  # alto
            else:
                base_risk = 2  # medio
        elif predicted_class in settings.HIGH_RISK_CLASSES:
            if confidence >= 0.7:
                base_risk = 3  # alto
            else:
                base_risk = 2  # medio
        else:
            if confidence >= 0.8:
                base_risk = 1  # bajo
            else:
                base_risk = 2  # medio

        # Adjust with ABCDE score
        if abcde_result and abcde_result.get("total_score"):
            abcde_total = abcde_result["total_score"]
            if abcde_total >= 0.7:
                base_risk = max(base_risk, 3)
            elif abcde_total >= 0.5:
                base_risk = max(base_risk, 2)

        risk_map = {1: "bajo", 2: "medio", 3: "alto", 4: "muy_alto"}
        return risk_map.get(min(base_risk, 4), "medio")

    def get_available_models(self) -> list:
        """List all available models and their status."""
        result = []
        for name, info in MODEL_REGISTRY.items():
            result.append({
                "name": name,
                "display_name": info["display_name"],
                "loaded": name in self.models,
                "input_size": info["input_size"],
                "context": info["context"],
                "auc_range": info["auc_range"],
                "latency_range": info["latency_range"]
            })
        return result


# Singleton instance
inference_engine = InferenceEngine()
