import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from siamese_model import INT8Siamese, Float32HardwareSurrogate

def load_int8_images(directory):
    tensors = []
    #.png converts to grayscale, and maps to [-128, 127] float
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            filepath = os.path.join(directory, filename)
            img = Image.open(filepath).convert('L')
            img_array = np.array(img, dtype=np.float32)
            img_centered = img_array - 128.0 
            tensors.append(torch.tensor(img_centered))
    return torch.stack(tensors) #[50, 8, 8]

print("Loading datasets...")
steve_data = load_int8_images("steve_8x8_processed")
alex_data = load_int8_images("alex_8x8_processed")

#generate 50 noisy images as a negative dataset
noise_data = torch.empty(50, 8, 8).uniform_(-128.0, 127.0)
print(f"Loaded {steve_data.shape[0]} Steves, {alex_data.shape[0]} Alexes, and 50 Noisy samples.")

#loss function for siamese network
def contrastive_loss(emb1, emb2, label, margin=50.0):
    #euclidean distance between the two 4-value embeddings
    distance = F.pairwise_distance(emb1.unsqueeze(0), emb2.unsqueeze(0))
    
    #1 (same face): pull embeddings together (minimize distance)
    #0 (different faces): push embeddings apart (maximize distance up to the margin)
    loss = label * torch.pow(distance, 2) + \
           (1 - label) * torch.pow(torch.clamp(margin - distance, min=0.0), 2)
    return loss.mean()

#declare model and optimizer
model = Float32HardwareSurrogate()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

epochs = 3000
pbar = tqdm(range(epochs), desc="Training Hardware Weights")

#training loop with dynamic pair generation
for epoch in pbar:
    optimizer.zero_grad()
    
    # generate a random pair of images and a label indicating if they are the same class (1) or different classes (0)
    is_same_class = np.random.rand() > 0.5
    
    idx1 = np.random.randint(0, 50)
    idx2 = np.random.randint(0, 50)
    
    if is_same_class:
        label = 1.0
        if np.random.rand() > 0.5:
            img1, img2 = steve_data[idx1], steve_data[idx2]
        else:
            img1, img2 = alex_data[idx1], alex_data[idx2]
    else:
        label = 0.0
        #3 types of different pairs (Steve-Alex, Steve-Noise, Alex-Noise)
        pair_type = np.random.randint(0, 3)
        if pair_type == 0:
            img1, img2 = steve_data[idx1], alex_data[idx2]
        elif pair_type == 1:
            img1, img2 = steve_data[idx1], noise_data[idx2]
        else:
            img1, img2 = alex_data[idx1], noise_data[idx2]
        
    emb1, emb2 = model(img1, img2)
    loss = contrastive_loss(emb1, emb2, label)
    
    loss.backward()
    optimizer.step()
    
    pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

#hardware export and verification
print("\n--- Training Complete ---")

# extract the trained weights, round them, and cast to int8
final_W_conv_int8 = torch.clamp(torch.round(model.W_conv.detach()), min=-128, max=127).to(torch.int8)
final_W_linear_int8 = torch.clamp(torch.round(model.W_linear.detach()), min=-128, max=127).to(torch.int8)

# instantiate model
hw_model = INT8Siamese()

#inject the trained weights into your custom class
hw_model.W_conv_int8 = final_W_conv_int8
hw_model.W_linear_int8 = final_W_linear_int8

#run a pure INT8 verification test
print("\nRunning Pure INT8 Hardware Verification:")

#two Steves, an Alex, and a Noise sample, cast them to strict int8
test_steve_1 = steve_data[np.random.randint(0, steve_data.shape[0])].to(torch.int8)
test_steve_2 = steve_data[np.random.randint(0, steve_data.shape[0])].to(torch.int8)
test_alex_1 = alex_data[np.random.randint(0, alex_data.shape[0])].to(torch.int8)
test_alex_2 = alex_data[np.random.randint(0, alex_data.shape[0])].to(torch.int8)
# NEW: Grab a random noise sample
test_noise_1 = noise_data[np.random.randint(0, noise_data.shape[0])].to(torch.int8)

#run them through your exact bit-shift logic
emb_s1 = hw_model.forward_once(test_steve_1)
emb_s2 = hw_model.forward_once(test_steve_2)
emb_a1 = hw_model.forward_once(test_alex_1)
emb_a2 = hw_model.forward_once(test_alex_2)
emb_n1 = hw_model.forward_once(test_noise_1) 

#calculate Euclidean distances manually
dist_steve_steve = torch.sqrt(torch.sum((emb_s1 - emb_s2) ** 2.0)).item()
dist_steve_alex = torch.sqrt(torch.sum((emb_s1 - emb_a1) ** 2.0)).item()
dist_alex_alex = torch.sqrt(torch.sum((emb_a1 - emb_a2) ** 2.0)).item()
#calculate distance to noise
dist_steve_noise = torch.sqrt(torch.sum((emb_s1 - emb_n1) ** 2.0)).item()
dist_alex_noise = torch.sqrt(torch.sum((emb_a1 - emb_n1) ** 2.0)).item()

print(f"Embeddings// steve1:{emb_s1}, steve2:{emb_s2}, alex1:{emb_a1}, alex2:{emb_a2}, noise:{emb_n1}")
print(f"Distance (Steve vs Steve): {dist_steve_steve:.2f} (Should be low)")
print(f"Distance (Steve vs Alex):  {dist_steve_alex:.2f} (Should be high)")
print(f"Distance (Alex vs Alex):  {dist_alex_alex:.2f} (Should be low)")
# NEW: Print noise verification
print(f"Distance (Steve vs Noise): {dist_steve_noise:.2f} (Should be high)")
print(f"Distance (Alex vs Noise):  {dist_alex_noise:.2f} (Should be high)")

#save model
torch.save(hw_model.W_conv_int8, "W_conv_int8.pth")
torch.save(hw_model.W_linear_int8, "W_linear_int8.pth")

#generate C compatible array
def generate_c_struct(tensor, name, struct_type):
    
    rows, cols = tensor.shape
    
    #start the struct and the 2D array
    c_str = f"{struct_type} {name} = {{\n    {{\n"
    
    for i in range(rows):
        #format the row values with proper spacing
        row_vals = [f"{val.item():4d}" for val in tensor[i]]
        c_str += f"        {{ {', '.join(row_vals)} }}"
        
        #add a comma to every row except the last one
        if i < rows - 1:
            c_str += ",\n"
        else:
            c_str += "\n"
            
    #close the 2D array and the struct
    c_str += "    }}\n};\n\n"
    return c_str

#slice 8x8 image into 4 patches
def format_image_for_hardware(img_8x8):

    patch_TL = img_8x8[0:4, 0:4].reshape(16)
    patch_TR = img_8x8[0:4, 4:8].reshape(16)
    patch_BL = img_8x8[4:8, 0:4].reshape(16)
    patch_BR = img_8x8[4:8, 4:8].reshape(16)
    return torch.stack([patch_TL, patch_TR, patch_BL, patch_BR], dim=0)

#export the weights
print("Exporting Weights to C format...")
with open("siamese_weights.c", "w") as f:
    f.write("// AUTOGENERATED INT8 WEIGHTS FOR CIMMETRY ACCELERATOR\n\n")
    f.write(generate_c_struct(final_W_conv_int8, "W_conv", "tensor16x4_t"))
    f.write(generate_c_struct(final_W_linear_int8, "W_linear", "tensor16x4_t"))

#export the test images
print("Exporting Test Images to C format...")
with open("test_images.c", "w") as f:
    f.write("// AUTOGENERATED INT8 TEST IMAGES (PRE-FORMATTED TO 4x16)\n\n")
    
    # Format and write the 4 images we used in the verification step
    img_s1_fmt = format_image_for_hardware(test_steve_1)
    f.write(generate_c_struct(img_s1_fmt, "test_steve_1", "tensor4x16_t"))
    
    img_s2_fmt = format_image_for_hardware(test_steve_2)
    f.write(generate_c_struct(img_s2_fmt, "test_steve_2", "tensor4x16_t"))
    
    img_a1_fmt = format_image_for_hardware(test_alex_1)
    f.write(generate_c_struct(img_a1_fmt, "test_alex_1", "tensor4x16_t"))
    
    img_a2_fmt = format_image_for_hardware(test_alex_2)
    f.write(generate_c_struct(img_a2_fmt, "test_alex_2", "tensor4x16_t"))

print("Done! Check 'siamese_weights.c' and 'test_images.c'.")