import numpy as np
from PIL import Image

from skimage.metrics import structural_similarity as ssim

def compute_ssim(
    image_path_1: str,
    image_path_2: str
):
    img_1 = Image.open(image_path_1).convert("RGB")
    img_2 = Image.open(image_path_2).convert("RGB")
    
    arr_1 = np.array(img_1)
    arr_2 = np.array(img_2)
    
    score, _ = ssim(arr_1, arr_2, full=True, multichannel=True)
    return score
