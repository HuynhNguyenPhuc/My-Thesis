import torch
from transformers import CLIPProcessor, CLIPModel
import os
from PIL import Image
import pandas as pd
import ast
from tqdm import tqdm

# -------------------- CLIP MODEL --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
model.eval()

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# -----------------------------------------------------


def compute_clip_similarity(image_path: str, text: str):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds

    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    similarity = (image_embeds @ text_embeds.T).item()
    return similarity


def compute_average_clip_per_shape(render_dir, caption):
    if not os.path.exists(render_dir):
        print(f"Render folder not found: {render_dir}, skipping.")
        return None, []

    image_files = [os.path.join(render_dir, f) for f in os.listdir(render_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"No images found in {render_dir}, skipping.")
        return None, []

    clip_scores = []

    for image_path in image_files:
        score = compute_clip_similarity(image_path, caption)
        clip_scores.append(score)

    if not clip_scores:
        return None, []

    avg_score = sum(clip_scores) / len(clip_scores)
    return avg_score, clip_scores


def compute_dataset_clip_scores(data_dir, test_dir):
    test_file_path = os.path.join(data_dir, "test.csv")
    test_df = pd.read_csv(test_file_path)


    results = []
    total_scores = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        sha256 = row["sha256"]
        render_dir = os.path.join(test_dir, 'renders', sha256)
        captions = ast.literal_eval(row["captions"])
        if not captions:
            print(f"No captions for {sha256}, skipping.")
            continue

        first_caption = captions[0]

        avg_score, clip_scores = compute_average_clip_per_shape(render_dir=render_dir, caption=first_caption)

        if avg_score is not None:
            results.append({"sha256": sha256, "average_clip_score": avg_score})
            total_scores.extend(clip_scores)

    overall_average = None
    if total_scores:
        overall_average = sum(total_scores) / len(total_scores)
    return overall_average
