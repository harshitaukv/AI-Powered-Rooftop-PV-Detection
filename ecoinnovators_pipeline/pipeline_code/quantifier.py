import numpy as np
def estimate_area_sqm(mask, image=None, ground_area_sqm=900.0):
    """
    Convert mask (2D numpy) to sqm.
    ground_area_sqm: the real-world area the image covers approx (default 30m x 30m = 900 sqm).
    pixel_area = ground_area_sqm / (W*H)
    """
    if mask is None:
        return 0.0
    H, W = mask.shape
    if H == 0 or W == 0:
        return 0.0
    pixel_area = float(ground_area_sqm) / float(W * H)
    area = float(mask.sum()) * pixel_area
    return area
