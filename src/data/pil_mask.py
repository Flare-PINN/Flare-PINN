"""
Polarity Inversion Line (PIL) Detection Module.

Efficiently detects PIL regions from Bz magnetograms using gradient-based methods.
PILs are critical for flare prediction as they indicate regions of magnetic stress.
"""
from __future__ import annotations
import numpy as np
from typing import Optional

# Sobel kernels for gradient computation
SOBEL_X = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]], dtype=np.float32)
SOBEL_Y = SOBEL_X.T.copy()


def _fast_conv2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Fast 2D convolution using scipy if available, else numpy fallback.
    
    This is ~100x faster than pure Python loop for typical image sizes.
    """
    try:
        from scipy.ndimage import convolve
        return convolve(img.astype(np.float32), kernel.astype(np.float32), mode='reflect')
    except ImportError:
        pass
    
    try:
        import cv2
        # OpenCV filter2D is even faster for large images
        return cv2.filter2D(img.astype(np.float32), -1, kernel, borderType=cv2.BORDER_REFLECT)
    except ImportError:
        pass
    
    # Numpy fallback using stride tricks (still faster than pure loops)
    from numpy.lib.stride_tricks import sliding_window_view
    
    H, W = img.shape
    kh, kw = kernel.shape
    
    # Pad image
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img.astype(np.float32), ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    
    # Create sliding windows
    windows = sliding_window_view(padded, (kh, kw))
    
    # Element-wise multiply and sum
    return np.einsum('ijkl,kl->ij', windows, kernel)


def compute_gradient_magnitude(Bz: np.ndarray) -> np.ndarray:
    """
    Compute |∇Bz| using Sobel operators.
    
    Args:
        Bz: [H, W] magnetic field
        
    Returns:
        gradient_mag: [H, W] magnitude of gradient
    """
    gx = _fast_conv2d(Bz, SOBEL_X)
    gy = _fast_conv2d(Bz, SOBEL_Y)
    return np.hypot(gx, gy)


def compute_pil_score(Bz: np.ndarray, gradient_mag: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute PIL score combining gradient magnitude and sign change proximity.
    
    High score regions are:
    1. High gradient magnitude (field changing rapidly)
    2. Near zero-crossing of Bz (polarity change)
    
    Args:
        Bz: [H, W] magnetic field
        gradient_mag: [H, W] pre-computed gradient (optional)
        
    Returns:
        pil_score: [H, W] PIL probability score
    """
    if gradient_mag is None:
        gradient_mag = compute_gradient_magnitude(Bz)
    
    # Zero-crossing proximity: high near Bz = 0
    # Use sigmoid-like decay from zero crossing
    bz_std = np.std(Bz[np.abs(Bz) > 1e-6]) if np.any(np.abs(Bz) > 1e-6) else 1.0
    zero_proximity = np.exp(-np.abs(Bz) / (0.5 * bz_std + 1e-8))
    
    # Combine: high gradient AND near zero crossing
    # Normalize gradient to [0, 1] range
    g_norm = gradient_mag / (np.percentile(gradient_mag, 99) + 1e-8)
    g_norm = np.clip(g_norm, 0, 1)
    
    # PIL score is product (both conditions must be met)
    pil_score = g_norm * zero_proximity
    
    return pil_score.astype(np.float32)


def pil_mask_from_bz(
    Bz: np.ndarray,
    top_percent: float = 15.0,
    close_radius_px: int = 2,
    do_skeletonize: bool = True,
    use_combined_score: bool = True
) -> np.ndarray:
    """
    Create binary PIL mask from Bz magnetogram.
    
    Algorithm:
    1. Compute gradient magnitude |∇Bz|
    2. Optionally combine with zero-crossing proximity
    3. Threshold to top N% of values
    4. Morphological closing to connect nearby regions
    5. Optional skeletonization for thin PIL lines
    
    Args:
        Bz: [H, W] vertical magnetic field
        top_percent: Threshold as top percentage of gradient values
        close_radius_px: Morphological closing radius (0 to disable)
        do_skeletonize: Whether to skeletonize the mask
        use_combined_score: Use combined gradient + zero-crossing score
        
    Returns:
        mask: [H, W] binary mask (uint8, values 0 or 1)
    """
    Bz = Bz.astype(np.float32)
    
    # Handle edge cases
    if Bz.size == 0 or np.all(Bz == 0):
        return np.zeros_like(Bz, dtype=np.uint8)
    
    # Compute gradient magnitude
    gradient_mag = compute_gradient_magnitude(Bz)
    
    # Optionally use combined PIL score
    if use_combined_score:
        score = compute_pil_score(Bz, gradient_mag)
    else:
        score = gradient_mag
    
    # Threshold to top percent
    if score.max() - score.min() < 1e-8:
        # Constant or near-constant field
        return np.zeros_like(Bz, dtype=np.uint8)
    
    thr = np.percentile(score, 100.0 - float(top_percent))
    mask = (score >= thr).astype(np.uint8)
    
    # Morphological closing to connect nearby PIL regions
    if close_radius_px > 0:
        try:
            from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
            str_el = generate_binary_structure(2, 1)
            mask = binary_dilation(mask, iterations=close_radius_px, structure=str_el)
            mask = binary_erosion(mask, iterations=close_radius_px, structure=str_el)
            mask = mask.astype(np.uint8)
        except ImportError:
            # Fallback: simple dilation using numpy
            from numpy.lib.stride_tricks import sliding_window_view
            if mask.shape[0] > 2 * close_radius_px and mask.shape[1] > 2 * close_radius_px:
                k = 2 * close_radius_px + 1
                padded = np.pad(mask, close_radius_px, mode='constant', constant_values=0)
                windows = sliding_window_view(padded, (k, k))
                mask = (windows.max(axis=(-2, -1)) > 0).astype(np.uint8)
    
    # Skeletonization for thin PIL lines
    if do_skeletonize:
        try:
            from skimage.morphology import skeletonize
            mask = skeletonize(mask > 0).astype(np.uint8)
        except ImportError:
            pass  # Skip if skimage not available
    
    return mask


def compute_pil_length(mask: np.ndarray) -> float:
    """
    Compute total PIL length in pixels.
    
    Useful as a feature for flare prediction (longer PILs = more stress).
    """
    return float(mask.sum())


def compute_pil_gradient_weighted_length(
    Bz: np.ndarray, 
    mask: np.ndarray
) -> float:
    """
    Compute gradient-weighted PIL length (GWPIL).
    
    Weights PIL pixels by local gradient magnitude.
    Strong gradient PILs are more likely to produce flares.
    """
    if mask.sum() == 0:
        return 0.0
    
    gradient_mag = compute_gradient_magnitude(Bz)
    weighted_length = (gradient_mag * mask.astype(np.float32)).sum()
    
    return float(weighted_length)


def compute_r_value(Bz: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute Schrijver's R-value: unsigned flux near PIL.
    
    R = log10(∫_PIL |Bz| ds)
    
    This is a key predictor of flare probability.
    """
    if mask.sum() == 0:
        return 0.0
    
    pil_flux = np.abs(Bz[mask > 0]).sum()
    
    if pil_flux <= 0:
        return 0.0
    
    return np.log10(pil_flux + 1e-10)

