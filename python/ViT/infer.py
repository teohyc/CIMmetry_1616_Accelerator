import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from model import INT8ViTSiamese

model = INT8ViTSiamese()
model.W_q = torch.load("W_q_int8.pth", weights_only=True) 
model.W_k = torch.load("W_k_int8.pth", weights_only=True)
model.W_v = torch.load("W_v_int8.pth", weights_only=True)
model.W_o = torch.load("W_o_int8.pth", weights_only=True)
model.W_up = torch.load("W_up_int8.pth", weights_only=True)
model.W_gate = torch.load("W_gate_int8.pth", weights_only=True)
model.W_down = torch.load("W_down_int8.pth", weights_only=True)

# Load 4 test images
def load_test_image(filepath):
    img = Image.open(filepath).convert('L')
    img_array = np.array(img, dtype=np.float32)
    img_centered = img_array - 128.0 
    return torch.tensor(img_centered).to(torch.int8)

test_steve_1 = load_test_image("steve_test1.png")
test_steve_2 = load_test_image("steve_test2.png")
test_alex_1 = load_test_image("alex_test1.png")
test_alex_2 = load_test_image("alex_test2.png")

# Run them through exact bit-shift logic
emb_s1 = model.forward_once(test_steve_1)
emb_s2 = model.forward_once(test_steve_2)
emb_a1 = model.forward_once(test_alex_1)
emb_a2 = model.forward_once(test_alex_2)

# Calculate Euclidean distances manually
dist_steve_steve = torch.sqrt(torch.sum((emb_s1 - emb_s2) ** 2.0)).item()
dist_steve_alex = torch.sqrt(torch.sum((emb_s1 - emb_a1) ** 2.0)).item()
dist_alex_alex = torch.sqrt(torch.sum((emb_a1 - emb_a2) ** 2.0)).item()

print(f"Distance (Steve vs Steve): {dist_steve_steve:.2f} (Should be low)")
print(f"Distance (Steve vs Alex):  {dist_steve_alex:.2f} (Should be high)")
print(f"Distance (Alex vs Alex):  {dist_alex_alex:.2f} (Should be low)")