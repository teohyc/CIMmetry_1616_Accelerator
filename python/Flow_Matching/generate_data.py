import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

from Flow_DiT_model import FlowDiT

@torch.no_grad()
def generate(model, steps=20):
    model.eval()
    x = torch.randn(1, 1, 8, 8).to(device) # Pure noise
    dt = 1.0 / steps
    
    for step in range(steps):
        t_val = step * dt
        t_tensor = torch.tensor([[t_val]]).to(device)
        v = model(x, t_tensor)
        x = x + v * dt #euler's method
    return x

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

alex_model = FlowDiT().to(device)
alex_model.load_state_dict(torch.load("alex_DiT.pth", map_location=device))

steve_model = FlowDiT().to(device)
steve_model.load_state_dict(torch.load("steve_DiT.pth", map_location=device))

alex_dir = "alex_8x8_processed"
steve_dir = "steve_8x8_processed"

#generate 46 new Alex and Steve images into their directories
for i in range(5, 51):

    #steve
    generated_tensor = generate(steve_model)

    #move back to CPU and convert the [-1, 1] float back to a [0, 255] integer array
    gen_array = generated_tensor[0, 0].cpu().numpy()
    gen_array_255 = np.clip((gen_array + 1.0) * 127.5, 0, 255).astype(np.uint8)

    # Save result
    output_img = Image.fromarray(gen_array_255, mode='L')
    output_path = f"{steve_dir}/steve_generated{i}.png"
    output_img.save(output_path)

    #alex
    generated_tensor = generate(alex_model)

    #move back to CPU and convert the [-1, 1] float back to a [0, 255] integer array
    gen_array = generated_tensor[0, 0].cpu().numpy()
    gen_array_255 = np.clip((gen_array + 1.0) * 127.5, 0, 255).astype(np.uint8)

    # Save result
    output_img = Image.fromarray(gen_array_255, mode='L')
    output_path = f"{alex_dir}/alex_generated{i}.png"
    output_img.save(output_path)