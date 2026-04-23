import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

from Flow_DiT_model import FlowDiT
from generate_data import generate

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = FlowDiT().to(device)
model.load_state_dict(torch.load("steve_DiT.pth", map_location=device)) #change model path to alex_DiT.pth to generate Alex images instead
generated_tensor = generate(model)

#move back to CPU and convert the [-1, 1] float back to a [0, 255] integer array
gen_array = generated_tensor[0, 0].cpu().numpy()
gen_array_255 = np.clip((gen_array + 1.0) * 127.5, 0, 255).astype(np.uint8)

# Save result
output_img = Image.fromarray(gen_array_255, mode='L')
output_path = "steve_test2.png" #change output path to generated_steve_experiment.png to generate Steve image instead
output_img.save(output_path)

print(f"Result at {output_path}")