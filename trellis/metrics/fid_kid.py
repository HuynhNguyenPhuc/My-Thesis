import os
from pathlib import Path
from itertools import chain

from PIL import Image
import numpy as np
from scipy import linalg

import torch
from torchvision import transforms

from .models.inception import InceptionV3

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x):
        return x


# -------------------- INCEPTION MODEL --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_dir = Path(os.path.realpath(__file__)).parent

# ---------------------------------------------------------


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, img_size, model_name, use_custom_ckpt=False):
        self.files = files
        self.img_size = img_size
        self.model_name = model_name
        self.use_custom_ckpt = use_custom_ckpt
        
        if model_name == "inceptionv3":
            if use_custom_ckpt:
                self.transforms = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ])
            else:
                self.transforms = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ])
        elif model_name == "dinov2":
            self.transforms = transforms.Compose([
                # transforms.Resize((img_size, img_size)),
                transforms.Resize((518, 518)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return img

def get_eval_loader(path, img_size, batch_size, model_name, use_custom_ckpt=False):
    def listdir(dname):
        fnames = list(
            chain(
                *[list(Path(dname).rglob("*." + ext)) for ext in ["png", "jpg", "jpeg", "JPG"]]
            )
        )
        return fnames

    files = listdir(path)
    ds = ImagePathDataset(files, img_size, model_name, use_custom_ckpt)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8
    )
    return dl

def frechet_distance(mu, cov, mu2, cov2):
    cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
    dist = np.sum((mu - mu2) ** 2) + np.trace(cov + cov2 - 2 * cc)
    return np.real(dist)
def kernel_score(mu1, cov1, mu2, cov2):
    diff_mu = mu1 - mu2
    diff_cov = cov1 - cov2
    kernel_value = np.exp(-0.5 * np.dot(diff_mu, diff_mu.T) / (2 * (np.trace(diff_cov) + 1e-8)))
    return kernel_value

@torch.no_grad()
def compute_fid_kid(paths, img_size=256, batch_size=50, model_name="inceptionv3", use_custom_ckpt=False):
    assert model_name in ['inceptionv3', 'dinov2'], "Model must be either 'inceptionv3' or 'dinov2'"

    if model_name == "inceptionv3":
        if use_custom_ckpt:
            ckpt_path = current_dir / "models" / "afhq_inception_v3.ckpt"
            ckpt = torch.load(ckpt_path, map_location="cpu")
            inception_model = InceptionV3(for_train=False)
            inception_model.load_state_dict(ckpt)
        else:
            inception_model = torch.hub.load("pytorch/vision", "inception_v3", pretrained=True)
            inception_model = inception_model.eval().to(device)

    elif model_name == "dinov2":
        dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        dinov2_model.eval().to(device)

    loaders = [get_eval_loader(path, img_size, batch_size, model_name, use_custom_ckpt) for path in paths]

    all_activations = []

    for loader in loaders:
        activations = []
        for x in tqdm(loader, total=len(loader)):
            if model_name == "inceptionv3":
                activation = inception_model(x.to(device))
            elif model_name == "dinov2":
                activation = dinov2_model(x.to(device))
            activations.append(activation)

        activations = torch.cat(activations, dim=0).cpu().detach().numpy()
        all_activations.append(activations)

    mu = [np.mean(act, axis=0) for act in all_activations]
    cov = [np.cov(act, rowvar=False) for act in all_activations]
    fid_value = frechet_distance(mu[0], cov[0], mu[1], cov[1])

    kid_value = kernel_score(mu[0], cov[0], mu[1], cov[1])

    return {'fid': fid_value, 'kid' : kid_value}


