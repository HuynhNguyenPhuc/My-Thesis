import os
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig, MediaResolution

# Load environment variables
load_dotenv()

# Set up Gemini API
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# Define the prompt for detailed description
PROMPT = """
These four images depict different angles (Front, Back, Left, and Right) of the same 3D object. Using all four perspectives, describe the object in detail, with equal emphasis on both its **visual characteristics** and its **capabilities, functionality, or actions**. Provide a concise yet descriptive paragraph covering the following aspects, synthesizing insights from all views:

**Visual Characteristics (Detailed):**

* **Shape and Form (Visual):** Describe the overall shape and form of the object *as seen in the thumbnail*. Is it geometric, organic, abstract, representational? Detail its structure, contours, and any prominent visual shapes.
* **Color and Material (Visual):** Describe the dominant colors, color palettes, and apparent material or surface textures *visually suggested by the thumbnail*. (e.g., shiny metal, rough wood, smooth plastic, vibrant fabric).
* **Visual Style and Aesthetic (Visual):** What is the overall visual style or aesthetic impression? Is it minimalist, ornate, realistic, stylized, futuristic, cartoonish, industrial, elegant, etc.? What kind of visual language does it employ?
* **Key Visual Features and Details (Visual):** Identify and describe any prominent visual features, distinctive elements, or notable visual details that stand out in the thumbnail.

**Capabilities, Functionality, and Actions (Balanced with Visuals):**

* **Functionality/Purpose (Inferred from Visuals):** What is the *likely* function, purpose, or intended use of this 3D object, *based on its visual design and cues in the thumbnail*? What kind of capability does it seem to represent?
* **Key Visual Cues for Function (Visual-Functional Link):** Point out specific *visual features* that strongly suggest its functionality or purpose. (e.g., wheels suggest mobility, blades suggest cutting, buttons suggest interaction, etc.) Explain how these visual cues relate to the inferred function.
* **Potential Actions/Interactions (Functional):** What actions or interactions might be associated with this object in a 3D context? (e.g., can it be opened, rotated, manipulated, used as a tool, sat upon, etc.?) Focus on the *actions* it enables or represents.
* **Context/Domain (Optional but helpful - Functional & Visual):** In what context or domain might this object typically be found or used? (e.g., furniture in a home, tools in a workshop, vehicles in a city, characters in a game). Consider both visual style and functionality to infer the domain.

Your description should thoroughly integrate both detailed visual observations and reasoned inferences about the object's function, ensuring both aspects are equally represented and linked where possible. Aim for a balanced paragraph that paints a picture of both *what it looks like* and *what it likely does or represents* based on the thumbnail.
"""

def _extract_captions(file, sha256, output_dir):
    """
    Generate detailed description and concise captions from 4 input images using Gemini Flash 2.0.
    
    Args:
        image_paths (list): List of paths to 4 images.
    
    Returns:
        None: Prints detailed description, one concise caption, and 10 brief captions.
    """
    
    image_paths = [os.path.join(output_dir, 'renders_caption', sha256, f'00{i}.png') for i in range(0, 4)]

    # Load 4 images from different angles (Front, Back, Left, and Right)
    views = [Image.open(p).convert("RGB") for p in image_paths]

    # Generate detailed description using Gemini
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=[PROMPT] + views,  # Pass the images
        config=GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            candidate_count=1,
            seed=None,
            media_resolution=MediaResolution.MEDIA_RESOLUTION_MEDIUM,
        )
    )
    detailed_description = response.text.strip()

    # Generate one concise caption
    concise_prompt = (
        "Carefully summarize in ONE caption aiming for **no more than 40 words** based on the following detailed description. "
        "Ensure it's concise and captures the essential information without any extra commentary or unnecessary details. "
        "Please avoid hallucination. "
        "Provide the caption in a simple, plain text format with bullets, numbering, extra text or special formatting. "
        f"Detailed description: {detailed_description}"
    )
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=[concise_prompt],
        config=GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            candidate_count=1,
            seed=None,
        )
    )
    concise_caption = response.text.strip()

    # Generate 10 progressively shorter captions
    brief_prompt = (
        "Now produce TEN versions of that caption, ONE per line, each more brief than the last. "
        "The first should be about 12 words (detailed). "
        "The last should be ≤5 words (super-brief), and it must still capture **every key object** from the original. "
        "You MUST NOT skip or generalize any object (e.g., 'structure' instead of 'building with solar panels'). "
        "Please avoid hallucination. "
        "No bullets, numbering, extra text or special formatting. "
        f"Caption: {concise_caption}"
    )
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=[brief_prompt],  # Pass only text
        config=GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            candidate_count=1,
            seed=None,
        )
    )
    brief_captions = response.text.strip()
    brief_captions = brief_captions.splitlines()
    brief_captions = [c for c in brief_captions if c.strip() != '']
    return {'sha256': sha256, 'captions': brief_captions}


if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=None)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    # get file list
    if not os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if 'captions' in metadata.columns:
            metadata = metadata[metadata['captions'].isna()]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
                
    print(f'Processing {len(metadata)} objects...')

    # process objects
    func = partial(_extract_captions, output_dir=opt.output_dir)
    captioned = dataset_utils.foreach_instance(metadata, opt.output_dir, func, max_workers=opt.max_workers, desc='Captioning')
    captioned.to_csv(os.path.join(opt.output_dir, f'captioned_{opt.rank}.csv'), index=False)