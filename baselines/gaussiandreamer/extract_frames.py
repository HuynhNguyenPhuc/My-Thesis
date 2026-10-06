from PIL import Image
import os

# Run from the repo root: python -m baselines.gaussiandreamer.extract_frames
gif_input_folder = 'outputs/gaussiandreamer_gifs'
frames_output_root = 'outputs/gaussiandreamer_frames'

def extract_uniform_frames(gif_path, output_folder):
    gif = Image.open(gif_path)
    total_frames = gif.n_frames
    interval = total_frames // 4

    for i in range(4):
        frame = i * interval
        if frame >= total_frames:
            frame = total_frames - 1

        gif.seek(frame)
        os.makedirs(output_folder, exist_ok=True)
        frame_path = os.path.join(output_folder, f"frame_{i}.png")
        gif.save(frame_path)
        print(f"Saved {frame_path}")

# Ensure root output folder exists
os.makedirs(frames_output_root, exist_ok=True)

# Loop through all .gif files
for filename in os.listdir(gif_input_folder):
    if filename.endswith('.gif'):
        gif_path = os.path.join(gif_input_folder, filename)
        name = os.path.splitext(filename)[0]  # remove ".gif"
        output_folder = os.path.join(frames_output_root, name)
        extract_uniform_frames(gif_path, output_folder)
