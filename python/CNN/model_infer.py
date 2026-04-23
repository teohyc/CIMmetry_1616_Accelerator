import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from siamese_model import INT8Siamese

#load model
model = INT8Siamese()
model.W_conv_int8 = torch.load("W_conv_int8.pth") #load quantized convolution weights
model.W_linear_int8 = torch.load("W_linear_int8.pth")

#load 4 test images (alex_test1.png, alex_test2.png, steve_test1.png, steve_test2.png) 
def load_test_image(filepath):
    img = Image.open(filepath).convert('L')
    img_array = np.array(img, dtype=np.float32)
    img_centered = img_array - 128.0 
    return torch.tensor(img_centered).to(torch.int8)

test_steve_1 = load_test_image("steve_test1.png")
test_steve_2 = load_test_image("steve_test2.png")
test_alex_1 = load_test_image("alex_test1.png")
test_alex_2 = load_test_image("alex_test2.png")

#run them through your exact bit-shift logic
emb_s1 = model.forward_once(test_steve_1)
emb_s2 = model.forward_once(test_steve_2)
emb_a1 = model.forward_once(test_alex_1)
emb_a2 = model.forward_once(test_alex_2)

#calculate Euclidean distances manually
dist_steve_steve = torch.sqrt(torch.sum((emb_s1 - emb_s2) ** 2.0)).item()
dist_steve_alex = torch.sqrt(torch.sum((emb_s1 - emb_a1) ** 2.0)).item()
dist_alex_alex = torch.sqrt(torch.sum((emb_a1 - emb_a2) ** 2.0)).item()

print(f"Distance (Steve vs Steve): {dist_steve_steve:.2f} (Should be low)")
print(f"Distance (Steve vs Alex):  {dist_steve_alex:.2f} (Should be high)")
print(f"Distance (Alex vs Alex):  {dist_alex_alex:.2f} (Should be low)")