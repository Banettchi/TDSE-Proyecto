"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.
Generates visual explanations showing which regions of the image
were most important for the model's classification decision.

Supports all three architectures (EfficientNet-B7, ResNet-50, ViT-Base).
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class GradCAM:
    """
    Generates Grad-CAM heatmaps for model interpretability.
    Maps the classification decision back to the image regions
    that drove it, enabling clinical validation via ABCDE framework.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.activations = output.detach()
            elif isinstance(output, tuple):
                self.activations = output[0].detach()

        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output[0], torch.Tensor):
                self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor,
                 target_class: Optional[int] = None) -> Tuple[np.ndarray, int, float]:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Preprocessed image tensor (1, C, H, W)
            target_class: Class to generate heatmap for (None = predicted class)

        Returns:
            (heatmap, predicted_class, confidence)
        """
        self.model.eval()
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        confidence = probs[0, target_class].item()

        # Backward pass for target class
        self.model.zero_grad()
        output[0, target_class].backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback: return a uniform heatmap
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            return np.ones((h, w), dtype=np.float32) * 0.5, target_class, confidence

        # Handle different tensor shapes (CNN vs Transformer)
        if len(self.activations.shape) == 4:
            # CNN: (B, C, H, W)
            weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
            cam = torch.sum(weights * self.activations, dim=1).squeeze()
        elif len(self.activations.shape) == 3:
            # Transformer: (B, N, D) where N = num_patches + 1 (cls token)
            gradients = self.gradients[0]  # (N, D)
            activations = self.activations[0]  # (N, D)

            weights = torch.mean(gradients, dim=0)
            cam = torch.sum(weights * activations, dim=-1)

            # Remove CLS token if present
            if cam.shape[0] > 1:
                cam = cam[1:]  # Remove CLS token

            # Reshape to 2D (assume square patch grid)
            num_patches = cam.shape[0]
            grid_size = int(np.sqrt(num_patches))
            if grid_size * grid_size == num_patches:
                cam = cam.reshape(grid_size, grid_size)
            else:
                cam = cam.reshape(1, -1)
        else:
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            return np.ones((h, w), dtype=np.float32) * 0.5, target_class, confidence

        # ReLU and normalize
        cam = F.relu(cam)
        if cam.max() > 0:
            cam = cam / cam.max()

        # Convert to numpy and resize to input dimensions
        heatmap = cam.cpu().numpy()
        heatmap = cv2.resize(heatmap, (input_tensor.shape[3], input_tensor.shape[2]))

        return heatmap, target_class, confidence

    def overlay_heatmap(self, image: np.ndarray, heatmap: np.ndarray,
                        alpha: float = 0.4, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Overlay the Grad-CAM heatmap on the original image.

        Args:
            image: Original BGR image
            heatmap: Normalized heatmap (0-1)
            alpha: Transparency of the overlay
            colormap: OpenCV colormap to use

        Returns:
            Overlay image (BGR)
        """
        # Resize heatmap to match image
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8),
            colormap
        )

        # Blend
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

        return overlay

    def generate_and_save(self, input_tensor: torch.Tensor,
                          original_image: np.ndarray,
                          save_path: str,
                          target_class: Optional[int] = None) -> Tuple[str, int, float]:
        """
        Generate Grad-CAM and save the overlay image.

        Returns:
            (save_path, predicted_class, confidence)
        """
        heatmap, pred_class, confidence = self.generate(input_tensor, target_class)
        overlay = self.overlay_heatmap(original_image, heatmap)

        cv2.imwrite(save_path, overlay)
        return save_path, pred_class, confidence
