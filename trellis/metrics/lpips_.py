# Install lpips library for Perceptual Similarity Metric
# ```
# pip install lpips
# ```

import torch
import lpips
from PIL import Image
from torchvision import transforms

# -------------------- LPIPS MODEL --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

lpips_model = lpips.LPIPS(net='alex').to(DEVICE)
lpips_model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# -----------------------------------------------------

def compute_lpips(
    image_path_1: str, 
    image_path_2: str
) -> float:
    img_1 = Image.open(image_path_1).convert("RGB")
    img_2 = Image.open(image_path_2).convert("RGB")
    
    tensor_1 = transform(img_1).unsqueeze(0).to(DEVICE)
    tensor_2 = transform(img_2).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        distance = lpips_model(tensor_1, tensor_2)
    
    return distance.item()

