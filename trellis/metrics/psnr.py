import math
from PIL import Image
import numpy as np

def compute_psnr(
    image_path_1: str, 
    image_path_2: str
) -> float:
    img_1 = Image.open(image_path_1).convert("RGB")
    img_2 = Image.open(image_path_2).convert("RGB")
    
    arr_1 = np.array(img_1, dtype=np.float32)
    arr_2 = np.array(img_2, dtype=np.float32)
    
    mse = np.mean((arr_1 - arr_2) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr_value = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr_value
