from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy import ndimage

def generate_overlay(image, mask, out_path, sample_id, radius_sqft=1200, confidence=0.0):
    """
    image: PIL.Image RGB
    mask: 2D numpy (same size as image or will be resized)
    out_path: where overlay is saved
    """
    im = image.copy().convert("RGBA")
    W, H = im.size
    mask_arr = mask
    try:
        mh, mw = mask_arr.shape
        if (mw, mh) != (W, H):
            mask_img = Image.fromarray((mask_arr * 255).astype('uint8')).resize((W, H))
            mask_arr = np.array(mask_img) > 128
    except Exception:
        mask_arr = np.zeros((H, W), dtype=bool)

    overlay = Image.new("RGBA", im.size, (0,0,0,0))
    overlay_px = overlay.load()
    for y in range(H):
        for x in range(W):
            if mask_arr[y, x]:
                overlay_px[x, y] = (255, 0, 0, 110) 

    combined = Image.alpha_composite(im, overlay)
    draw = ImageDraw.Draw(combined)
    txt = f"ID:{sample_id}  radius:{radius_sqft}sqft  conf:{confidence:.2f}"
    try:
        font = ImageFont.load_default()
    except:
        font = None
    draw.text((8, 8), txt, fill=(255,255,255), font=font)
    combined.convert("RGB").save(out_path)
    return out_path

def qc_check(image, mask):
    """
    Simple QC rules:
    - Exclude if too many bright pixels (clouds) or too many dark pixels (shadow)
    - Exclude if blurry (low Laplacian variance)
    """
    gray = image.convert("L")
    arr = np.array(gray).astype(float)
    lap = ndimage.laplace(arr)
    var = float(lap.var())
    bright_prop = float((arr > 245).mean())
    dark_prop = float((arr < 40).mean())
    if bright_prop > 0.15:
        return "NOT_VERIFIABLE"
    if var < 20:
        return "NOT_VERIFIABLE"
    if dark_prop > 0.3:
        return "NOT_VERIFIABLE"
    return "VERIFIABLE"

