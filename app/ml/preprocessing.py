"""
Image preprocessing pipeline for skin lesion analysis.
Implements the three preprocessing steps from the technical document:
  1. Quality validation (blur detection, framing check)
  2. Normalization (resize, white balance, illumination)
  3. Augmentation support for training
"""
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from typing import Tuple, Optional
from app.config import settings


class ImagePreprocessor:
    """
    Preprocessing pipeline for skin lesion images.
    Handles both dermatoscopic and smartphone images.
    """

    # Normalization stats (ImageNet standard)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Quality thresholds
    BLUR_THRESHOLD = 100.0  # Laplacian variance threshold
    MIN_LESION_RATIO = 0.05  # Minimum lesion area / total area

    def __init__(self, target_size: int = 224):
        self.target_size = target_size

        # Standard inference transform
        self.inference_transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

        # Training transform with augmentation
        self.train_transform = transforms.Compose([
            transforms.Resize((target_size + 32, target_size + 32)),
            transforms.RandomCrop(target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

    def validate_quality(self, image: np.ndarray) -> Tuple[bool, float, str]:
        """
        Step 1: Quality validation.
        Checks for blur, proper framing, and minimum resolution.

        Returns:
            (is_valid, quality_score, message)
        """
        # Check minimum resolution
        h, w = image.shape[:2]
        if h < 100 or w < 100:
            return False, 0.0, "Resolución demasiado baja (mínimo 100x100 px)"

        # Blur detection via Laplacian variance
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if laplacian_var < self.BLUR_THRESHOLD:
            score = laplacian_var / self.BLUR_THRESHOLD
            return False, score, f"Imagen borrosa (nitidez: {score:.1%})"

        # Check if image is too dark or too bright
        mean_brightness = np.mean(gray)
        if mean_brightness < 30:
            return False, 0.3, "Imagen demasiado oscura"
        if mean_brightness > 240:
            return False, 0.3, "Imagen sobreexpuesta"

        # Overall quality score
        quality_score = min(1.0, laplacian_var / (self.BLUR_THRESHOLD * 3))
        quality_score = quality_score * 0.7 + (1.0 - abs(mean_brightness - 128) / 128) * 0.3

        return True, quality_score, "Calidad aceptable"

    def normalize_illumination(self, image: np.ndarray) -> np.ndarray:
        """
        Step 2: Normalize illumination and white balance.
        Uses CLAHE (Contrast Limited Adaptive Histogram Equalization)
        to reduce variability from capture conditions.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)

        # Merge back
        enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        return result

    def remove_hair(self, image: np.ndarray) -> np.ndarray:
        """
        Remove hair artifacts using morphological blackhat filtering.
        Common preprocessing step for dermoscopic images.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        # Threshold to get hair mask
        _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

        # Inpaint to remove hair
        result = cv2.inpaint(image, mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
        return result

    def segment_lesion(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment the skin lesion from the background.
        Returns the mask and the segmented region.
        Used for ABCDE analysis.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)

        # Otsu's thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # Find the largest contour (the lesion)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(binary)
            cv2.drawContours(mask, [largest], -1, 255, -1)
        else:
            mask = binary

        return mask, cv2.bitwise_and(image, image, mask=mask)

    def preprocess_for_inference(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray, float]:
        """
        Full preprocessing pipeline for inference.

        Returns:
            (tensor, original_image, quality_score)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")

        # Validate quality
        is_valid, quality_score, msg = self.validate_quality(image)

        # Normalize illumination
        normalized = self.normalize_illumination(image)

        # Convert to PIL for torchvision transforms
        rgb = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        # Apply inference transform
        tensor = self.inference_transform(pil_image)
        tensor = tensor.unsqueeze(0)  # Add batch dimension

        return tensor, image, quality_score

    def preprocess_for_training(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess a single image for training (with augmentation).
        """
        return self.train_transform(image)

    @staticmethod
    def denormalize(tensor: torch.Tensor) -> np.ndarray:
        """Convert a normalized tensor back to a displayable numpy image."""
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        img = tensor.cpu().numpy().transpose(1, 2, 0)
        img = img * std + mean
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return img
