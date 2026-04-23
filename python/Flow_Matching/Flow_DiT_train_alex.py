import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from tqdm import tqdm

from Flow_DiT_model import FlowDiT

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

input_dir = "alex_8x8_processed"
tensors = []

for i in range(1, 5):
    filepath = os.path.join(input_dir, f"alex{i}_8x8.png")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing {filepath}")
    
    img = Image.open(filepath).convert('L')
    img_array = np.array(img, dtype=np.float32)

    #normalize
    img_normalized = (img_array / 127.5) - 1.0

    #convert to pytorch tensor
    tensors.append(torch.tensor(img_normalized).unsqueeze(0))

alex_data = torch.stack(tensors).to(device)
print(f"Successfully loaded 4 images. Batch shape: {alex_data.shape}")

#training loop
model = FlowDiT().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 2000
batch_size = 4

print("Starting training...")
model.train()

pbar = tqdm(range(epochs), desc="Training DiT")
for epoch in pbar:
    optimizer.zero_grad()

    x_data = alex_data
    x_noise = torch.randn_like(x_data)

    #sample random time from [0,1]
    t = torch.rand(batch_size, 1).to(device)
    t_img = t.view(batch_size, 1, 1, 1)

    #linear interpolation
    x_t = t_img * x_data + (1 - t_img) * x_noise

    #target velocity
    v_target = x_data - x_noise #dx_dt = x_data - x_noise 

    #predict and loss
    v_pred = model(x_t, t)
    loss = torch.nn.functional.mse_loss(v_pred, v_target)

    loss.backward()
    optimizer.step()

    pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

    if epoch % 500 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

@torch.no_grad()
def generate_steve(model, steps=20):
    model.eval()
    x = torch.randn(1, 1, 8, 8).to(device) # Pure noise
    dt = 1.0 / steps
    
    for step in range(steps):
        t_val = step * dt
        t_tensor = torch.tensor([[t_val]]).to(device)
        v = model(x, t_tensor)
        x = x + v * dt #euler's method
    return x

print("Generating new Alex...")
generated_tensor = generate_steve(model)

#move back to CPU and convert the [-1, 1] float back to a [0, 255] integer array
gen_array = generated_tensor[0, 0].cpu().numpy()
gen_array_255 = np.clip((gen_array + 1.0) * 127.5, 0, 255).astype(np.uint8)

# Save result
output_img = Image.fromarray(gen_array_255, mode='L')
output_path = "alex_8x8_processed/generated_alex_experiment.png"
output_img.save(output_path)

print(f"Result at {output_path}")

#save model
torch.save(model.state_dict(), "alex_DiT.pth")