import os
import torch
from PIL import Image
import numpy as np

input_dir = "steve_image"
output_dir = "steve_8x8_processed"

os.makedirs(output_dir, exist_ok=True)

#loop through images
for i in range(1, 5):
    filename = f"steve{i}.jpg"
    filepath = os.path.join(input_dir, filename)

    if not os.path.exists(filepath):
        print(f"Warning: Could not find {filepath}")
        continue

    #open image and convert to grayscale
    img = Image.open(filepath).convert("L")

    #resize to 8x8
    img_8x8 = img.resize((8, 8), Image.Resampling.NEAREST) #maintain hard contrast line

    #save
    save_path = os.path.join(output_dir, f"steve{i}_8x8.png")
    img_8x8.save(save_path)

input_dir = "alex_image"
output_dir = "alex_8x8_processed"

os.makedirs(output_dir, exist_ok=True)

#loop through images
for i in range(1, 5):
    filename = f"alex{i}.jpg"
    filepath = os.path.join(input_dir, filename)

    if not os.path.exists(filepath):
        print(f"Warning: Could not find {filepath}")
        continue

    #open image and convert to grayscale
    img = Image.open(filepath).convert("L")

    #resize to 8x8
    img_8x8 = img.resize((8, 8), Image.Resampling.NEAREST) #maintain hard contrast line

    #save
    save_path = os.path.join(output_dir, f"alex{i}_8x8.png")
    img_8x8.save(save_path)