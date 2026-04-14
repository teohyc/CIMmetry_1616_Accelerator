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
print(f"Loaded {steve_data.shape[0]} Steves and {alex_data.shape[0]} Alexes.")

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

epochs = 1000
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
        img1, img2 = steve_data[idx1], alex_data[idx2]
        
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

#two Steves and an Alex, cast them to strict int8
test_steve_1 = steve_data[np.random.randint(0, steve_data.shape[0])].to(torch.int8)
test_steve_2 = steve_data[np.random.randint(0, steve_data.shape[0])].to(torch.int8)
test_alex_1 = alex_data[np.random.randint(0, alex_data.shape[0])].to(torch.int8)
test_alex_2 = alex_data[np.random.randint(0, alex_data.shape[0])].to(torch.int8)

#run them through your exact bit-shift logic
emb_s1 = hw_model.forward_once(test_steve_1)
emb_s2 = hw_model.forward_once(test_steve_2)
emb_a1 = hw_model.forward_once(test_alex_1)
emb_a2 = hw_model.forward_once(test_alex_2)

#calculate Euclidean distances manually
dist_steve_steve = torch.sqrt(torch.sum((emb_s1 - emb_s2) ** 2.0)).item()
dist_steve_alex = torch.sqrt(torch.sum((emb_s1 - emb_a1) ** 2.0)).item()
dist_alex_alex = torch.sqrt(torch.sum((emb_a1 - emb_a2) ** 2.0)).item()

print(f"Distance (Steve vs Steve): {dist_steve_steve:.2f} (Should be low)")
print(f"Distance (Steve vs Alex):  {dist_steve_alex:.2f} (Should be high)")
print(f"Distance (Alex vs Alex):  {dist_alex_alex:.2f} (Should be low)")

#save model
torch.save(hw_model.W_conv_int8, "W_conv_int8.pth")
torch.save(hw_model.W_linear_int8, "W_linear_int8.pth")

#write weigths to text files in C array format for testbench
def save_weights_to_c_array(weights, filename):
    with open(filename, 'w') as f:
        f.write("int8_t " + filename.split('.')[0] + "[] = {")
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                f.write(f"{weights[i, j].item()}, ")
        f.write("};\n") 

save_weights_to_c_array(final_W_conv_int8, "W_conv_int8.c")
save_weights_to_c_array(final_W_linear_int8, "W_linear_int8.c")


