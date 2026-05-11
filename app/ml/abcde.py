"""
ABCDE Rule Analysis for skin lesion evaluation.
Implements computational analysis of the clinical ABCDE criteria:
  A - Asymmetry: Area moments and inertia axes
  B - Border: Contour irregularity via Fourier descriptors
  C - Color: Variance in HSV/L*a*b* color spaces
  D - Diameter: Estimated physical size
  E - Evolution: Temporal comparison between sequential images
"""
import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from scipy import ndimage


class ABCDEAnalyzer:
    """
    Computes ABCDE scores from a skin lesion image.
    Each score is normalized to [0, 1] where higher = more suspicious.
    """

    # ABCDE weights for total score (clinical consensus)
    WEIGHTS = {
        'A': 1.3,  # Asymmetry
        'B': 1.0,  # Border
        'C': 1.0,  # Color
        'D': 0.5,  # Diameter
        'E': 1.0   # Evolution
    }

    # Maximum total score (without evolution)
    MAX_SCORE_NO_EVOLUTION = 1.3 + 1.0 + 1.0 + 0.5  # 3.8

    def analyze(self, image: np.ndarray, mask: Optional[np.ndarray] = None,
                previous_image: Optional[np.ndarray] = None,
                previous_mask: Optional[np.ndarray] = None) -> Dict:
        """
        Perform full ABCDE analysis on a lesion image.

        Args:
            image: BGR image of the lesion
            mask: Binary mask of the lesion (will be computed if None)
            previous_image: Previous image of same lesion for evolution analysis
            previous_mask: Previous mask for evolution analysis

        Returns:
            Dictionary with individual scores and total
        """
        if mask is None:
            mask = self._segment(image)

        # Ensure mask is binary
        mask = (mask > 127).astype(np.uint8) * 255

        a_score, a_details = self._analyze_asymmetry(mask)
        b_score, b_details = self._analyze_border(mask)
        c_score, c_details = self._analyze_color(image, mask)
        d_score, d_details = self._analyze_diameter(mask, image.shape)

        e_score = None
        e_details = {}
        if previous_image is not None and previous_mask is not None:
            e_score, e_details = self._analyze_evolution(
                image, mask, previous_image, previous_mask
            )

        # Calculate weighted total
        total = (
            a_score * self.WEIGHTS['A'] +
            b_score * self.WEIGHTS['B'] +
            c_score * self.WEIGHTS['C'] +
            d_score * self.WEIGHTS['D']
        )

        max_possible = self.MAX_SCORE_NO_EVOLUTION
        if e_score is not None:
            total += e_score * self.WEIGHTS['E']
            max_possible += self.WEIGHTS['E']

        # Normalize to 0-1
        total_normalized = total / max_possible

        return {
            'asymmetry': round(a_score, 4),
            'border': round(b_score, 4),
            'color': round(c_score, 4),
            'diameter': round(d_details.get('diameter_mm', 0), 2),
            'diameter_score': round(d_score, 4),
            'evolution': round(e_score, 4) if e_score is not None else None,
            'total_score': round(total_normalized, 4),
            'details': {
                'asymmetry': a_details,
                'border': b_details,
                'color': c_details,
                'diameter': d_details,
                'evolution': e_details
            }
        }

    def _segment(self, image: np.ndarray) -> np.ndarray:
        """Simple segmentation fallback using Otsu's method."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(binary)
            cv2.drawContours(mask, [largest], -1, 255, -1)
            return mask
        return binary

    def _analyze_asymmetry(self, mask: np.ndarray) -> Tuple[float, dict]:
        """
        Asymmetry (A): Quantified via area moments and inertia axes.
        Compares the lesion halves along major and minor axes.
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, {'method': 'moments', 'axis1_diff': 0, 'axis2_diff': 0}

        contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(contour)

        if moments['m00'] == 0:
            return 0.0, {'method': 'moments', 'axis1_diff': 0, 'axis2_diff': 0}

        # Centroid
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])

        # Calculate orientation angle
        if moments['mu20'] - moments['mu02'] != 0:
            angle = 0.5 * np.arctan2(2 * moments['mu11'], moments['mu20'] - moments['mu02'])
        else:
            angle = 0

        # Rotate mask to align with principal axis
        h, w = mask.shape
        M = cv2.getRotationMatrix2D((cx, cy), np.degrees(angle), 1)
        rotated = cv2.warpAffine(mask, M, (w, h))

        # Split along horizontal axis (major axis after rotation)
        top_half = rotated[:cy, :]
        bottom_half = rotated[cy:, :]

        # Flip bottom half for comparison
        bottom_flipped = cv2.flip(bottom_half, 0)

        # Resize to same dimensions for comparison
        min_h = min(top_half.shape[0], bottom_flipped.shape[0])
        min_w = min(top_half.shape[1], bottom_flipped.shape[1])

        if min_h == 0 or min_w == 0:
            return 0.0, {'method': 'moments', 'axis1_diff': 0, 'axis2_diff': 0}

        top_crop = top_half[-min_h:, :min_w]
        bottom_crop = bottom_flipped[:min_h, :min_w]

        # XOR to find asymmetric areas
        diff_h = cv2.bitwise_xor(top_crop, bottom_crop)
        total_area = cv2.countNonZero(rotated)
        asym_h = cv2.countNonZero(diff_h) / max(total_area, 1)

        # Repeat for vertical axis
        left_half = rotated[:, :cx]
        right_half = rotated[:, cx:]
        right_flipped = cv2.flip(right_half, 1)

        min_h2 = min(left_half.shape[0], right_flipped.shape[0])
        min_w2 = min(left_half.shape[1], right_flipped.shape[1])

        if min_h2 == 0 or min_w2 == 0:
            asym_v = 0.0
        else:
            left_crop = left_half[:min_h2, -min_w2:]
            right_crop = right_flipped[:min_h2, :min_w2]
            diff_v = cv2.bitwise_xor(left_crop, right_crop)
            asym_v = cv2.countNonZero(diff_v) / max(total_area, 1)

        # Combined asymmetry score (average of both axes)
        score = min(1.0, (asym_h + asym_v) / 2)

        return score, {
            'method': 'moment_axes',
            'axis1_diff': round(asym_h, 4),
            'axis2_diff': round(asym_v, 4),
            'orientation_deg': round(np.degrees(angle), 2)
        }

    def _analyze_border(self, mask: np.ndarray) -> Tuple[float, dict]:
        """
        Border (B): Irregularity via Fourier descriptors of the contour.
        More high-frequency components = more irregular border.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return 0.0, {'method': 'fourier', 'irregularity': 0, 'compactness': 0}

        contour = max(contours, key=cv2.contourArea)

        # Compactness (circularity)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0.0, {'method': 'fourier', 'irregularity': 0, 'compactness': 0}

        compactness = (4 * np.pi * area) / (perimeter * perimeter)

        # Fourier descriptors
        contour_complex = contour[:, 0, 0] + 1j * contour[:, 0, 1]
        fourier = np.fft.fft(contour_complex)

        # Normalize by DC component
        if abs(fourier[0]) > 0:
            fourier_norm = np.abs(fourier) / abs(fourier[0])
        else:
            fourier_norm = np.abs(fourier)

        # High frequency energy ratio (irregularity indicator)
        n = len(fourier_norm)
        low_freq = np.sum(fourier_norm[1:max(2, n // 8)])
        high_freq = np.sum(fourier_norm[n // 8:n // 2])
        total_energy = low_freq + high_freq

        if total_energy > 0:
            irregularity = high_freq / total_energy
        else:
            irregularity = 0

        # Combine compactness and Fourier irregularity
        border_score = min(1.0, (1 - compactness) * 0.5 + irregularity * 0.5)

        return border_score, {
            'method': 'fourier_descriptors',
            'irregularity': round(irregularity, 4),
            'compactness': round(compactness, 4),
            'perimeter_px': round(perimeter, 1),
            'area_px': int(area)
        }

    def _analyze_color(self, image: np.ndarray, mask: np.ndarray) -> Tuple[float, dict]:
        """
        Color (C): Quantified via variance in HSV and L*a*b* color spaces.
        Multiple distinct colors in a lesion increase malignancy suspicion.
        """
        # Extract lesion pixels
        lesion_pixels = image[mask > 0]
        if len(lesion_pixels) == 0:
            return 0.0, {'method': 'color_variance', 'num_colors': 0}

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_pixels = hsv[mask > 0]

        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab_pixels = lab[mask > 0]

        # HSV variance
        h_var = np.var(hsv_pixels[:, 0].astype(float))
        s_var = np.var(hsv_pixels[:, 1].astype(float))
        v_var = np.var(hsv_pixels[:, 2].astype(float))

        # LAB variance
        l_var = np.var(lab_pixels[:, 0].astype(float))
        a_var = np.var(lab_pixels[:, 1].astype(float))
        b_var = np.var(lab_pixels[:, 2].astype(float))

        # Count distinct color clusters (simplified via histogram)
        # Quantize HSV to count distinct colors
        h_quant = (hsv_pixels[:, 0] / 30).astype(int)  # 12 hue bins
        s_quant = (hsv_pixels[:, 1] / 85).astype(int)   # 3 sat bins
        v_quant = (hsv_pixels[:, 2] / 85).astype(int)   # 3 val bins

        color_codes = h_quant * 9 + s_quant * 3 + v_quant
        unique_colors = len(np.unique(color_codes))

        # Normalize color variance
        # Higher variance = more color diversity = higher suspicion
        max_hsv_var = 180 * 180  # Max possible H variance
        hsv_score = min(1.0, (h_var / max_hsv_var) * 3)

        max_lab_var = 128 * 128  # Approximate max a/b variance
        lab_score = min(1.0, ((a_var + b_var) / (2 * max_lab_var)) * 3)

        # Color count score (>3 distinct colors is suspicious)
        color_count_score = min(1.0, unique_colors / 6)

        # Combined
        color_score = min(1.0, hsv_score * 0.3 + lab_score * 0.3 + color_count_score * 0.4)

        return color_score, {
            'method': 'hsv_lab_variance',
            'hsv_variance': {
                'h': round(h_var, 2),
                's': round(s_var, 2),
                'v': round(v_var, 2)
            },
            'lab_variance': {
                'l': round(l_var, 2),
                'a': round(a_var, 2),
                'b': round(b_var, 2)
            },
            'num_colors': int(unique_colors)
        }

    def _analyze_diameter(self, mask: np.ndarray, image_shape: tuple) -> Tuple[float, dict]:
        """
        Diameter (D): Estimated physical size.
        >6mm is a clinical threshold for concern.
        Estimates real size assuming standard capture distance.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, {'diameter_mm': 0, 'diameter_px': 0}

        contour = max(contours, key=cv2.contourArea)

        # Minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        diameter_px = radius * 2

        # Estimate physical size (assumption: 15cm capture distance, ~50px/mm for smartphone)
        # This is a rough estimation - in production, calibration would be required
        image_diag_px = np.sqrt(image_shape[0] ** 2 + image_shape[1] ** 2)
        px_per_mm = image_diag_px / 150  # Assume ~150mm field of view
        diameter_mm = diameter_px / max(px_per_mm, 1)

        # Score: >6mm is concerning
        if diameter_mm >= 6:
            score = min(1.0, diameter_mm / 12)
        else:
            score = diameter_mm / 12

        return score, {
            'diameter_mm': round(diameter_mm, 2),
            'diameter_px': round(diameter_px, 1),
            'threshold_mm': 6.0,
            'exceeds_threshold': diameter_mm >= 6.0
        }

    def _analyze_evolution(self, current_image: np.ndarray, current_mask: np.ndarray,
                           previous_image: np.ndarray, previous_mask: np.ndarray) -> Tuple[float, dict]:
        """
        Evolution (E): Temporal comparison between sequential images.
        Measures changes in area, shape, and color over time.
        """
        # Resize to same dimensions
        h, w = current_image.shape[:2]
        prev_resized = cv2.resize(previous_image, (w, h))
        prev_mask_resized = cv2.resize(previous_mask, (w, h))

        # Area change
        current_area = cv2.countNonZero(current_mask)
        prev_area = cv2.countNonZero(prev_mask_resized)

        if prev_area > 0:
            area_change = abs(current_area - prev_area) / prev_area
        else:
            area_change = 0

        # Shape change (contour matching)
        c1, _ = cv2.findContours(current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c2, _ = cv2.findContours(prev_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        shape_diff = 0
        if c1 and c2:
            shape_diff = cv2.matchShapes(
                max(c1, key=cv2.contourArea),
                max(c2, key=cv2.contourArea),
                cv2.CONTOURS_MATCH_I2, 0
            )

        # Color change in LAB space
        current_lab = cv2.cvtColor(current_image, cv2.COLOR_BGR2LAB)
        prev_lab = cv2.cvtColor(prev_resized, cv2.COLOR_BGR2LAB)

        current_pixels = current_lab[current_mask > 0].mean(axis=0) if cv2.countNonZero(current_mask) > 0 else np.zeros(3)
        prev_pixels = prev_lab[prev_mask_resized > 0].mean(axis=0) if cv2.countNonZero(prev_mask_resized) > 0 else np.zeros(3)

        color_change = np.linalg.norm(current_pixels - prev_pixels) / 255

        # Combined evolution score
        evolution_score = min(1.0,
                             area_change * 0.4 +
                             min(1.0, shape_diff) * 0.3 +
                             color_change * 0.3)

        return evolution_score, {
            'area_change_pct': round(area_change * 100, 2),
            'shape_difference': round(shape_diff, 4),
            'color_change': round(color_change, 4),
            'area_grew': current_area > prev_area
        }
