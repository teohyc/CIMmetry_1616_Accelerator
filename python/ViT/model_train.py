import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import HardwareViTSurrogate


def load_int8_images(directory, count=50):
    tensors = []
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if filename.endswith(".png"):
                filepath = os.path.join(directory, filename)
                img = Image.open(filepath).convert('L')
                img_array = np.array(img, dtype=np.float32)
                tensors.append(torch.tensor(img_array - 128.0))
    # Fallback to random noise if directory missing
    if len(tensors) == 0:
        print(f"Warning: '{directory}' not found. Generating dummy data.")
        return torch.empty(count, 8, 8).uniform_(-128.0, 127.0)
    return torch.stack(tensors)

def contrastive_loss(emb1, emb2, label, margin=3.0):
    distance = F.pairwise_distance(emb1, emb2)
    loss = label * torch.pow(distance, 2) + \
           (1 - label) * torch.pow(torch.clamp(margin - distance, min=0.0), 2)
    return loss.mean()

# ==============================================================================
#  TRAINING PIPELINE
# ==============================================================================
if __name__ == "__main__":
    print("[*] Loading datasets...")
    steve_data = load_int8_images("steve_8x8_processed")
    alex_data = load_int8_images("alex_8x8_processed")
    noise_data = torch.empty(50, 8, 8).uniform_(-128.0, 127.0)
    
    # Pre-generate a static validation set to track accuracy cleanly
    val_pairs_img1, val_pairs_img2, val_labels = [], [], []
    for _ in range(100):
        is_same = np.random.rand() > 0.5
        val_labels.append(1.0 if is_same else 0.0)
        idx1, idx2 = np.random.randint(0, 50, 2)
        if is_same:
            if np.random.rand() > 0.5:
                val_pairs_img1.append(steve_data[idx1])
                val_pairs_img2.append(steve_data[idx2])
            else:
                val_pairs_img1.append(alex_data[idx1])
                val_pairs_img2.append(alex_data[idx2])
        else:
            val_pairs_img1.append(steve_data[idx1])
            val_pairs_img2.append(alex_data[idx2])
            
    val_t1 = torch.stack(val_pairs_img1)
    val_t2 = torch.stack(val_pairs_img2)
    val_lbls = torch.tensor(val_labels)

    model = HardwareViTSurrogate()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    epochs = 3000
    
    train_losses = []
    val_accuracies = []

    print("[*] Starting QAT for ViT-SwiGLU...")
    pbar = tqdm(range(epochs))
    
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        
        # Dynamic Batch Generation (Batch Size = 64)
        batch_img1, batch_img2, batch_lbl = [], [], []
        for _ in range(64):
            is_same = np.random.rand() > 0.5
            idx1, idx2 = np.random.randint(0, 50, 2)
            batch_lbl.append(1.0 if is_same else 0.0)
            
            if is_same:
                src = steve_data if np.random.rand() > 0.5 else alex_data
                batch_img1.append(src[idx1])
                batch_img2.append(src[idx2])
            else:
                p_type = np.random.randint(0, 3)
                if p_type == 0:
                    batch_img1.append(steve_data[idx1])
                    batch_img2.append(alex_data[idx2])
                elif p_type == 1:
                    batch_img1.append(steve_data[idx1])
                    batch_img2.append(noise_data[idx2])
                else:
                    batch_img1.append(alex_data[idx1])
                    batch_img2.append(noise_data[idx2])
                    
        b_t1 = torch.stack(batch_img1)
        b_t2 = torch.stack(batch_img2)
        b_lbl = torch.tensor(batch_lbl)

        # Forward & Backprop
        emb1, emb2 = model(b_t1, b_t2)
        loss = contrastive_loss(emb1, emb2, b_lbl)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # Validation Eval (Every 50 epochs for speed)
        if epoch % 50 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                v_emb1, v_emb2 = model(val_t1, val_t2)
                dists = F.pairwise_distance(v_emb1, v_emb2)
                
                # Dynamic Thresholding for Accuracy
                best_acc = 0
                for th in np.arange(0.1, 2.5, 0.1):
                    preds = (dists < th).float()
                    acc = (preds == val_lbls).float().mean().item()
                    if acc > best_acc:
                        best_acc = acc
                val_accuracies.append(best_acc)
                
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Val Acc": f"{val_accuracies[-1]*100:.1f}%"})

    # ==============================================================================
    #  EXPORT & PLOTTING
    # ==============================================================================
    print("\n[*] Training Complete. Generating Graphs & Firmware...")
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, color='royalblue', alpha=0.7)
    plt.title('Contrastive Loss')
    plt.xlabel('Iterations')
    
    plt.subplot(1, 2, 2)
    # Stretch val_accuracies across the x-axis for plotting
    x_vals = np.linspace(0, epochs, len(val_accuracies))
    plt.plot(x_vals, val_accuracies, color='crimson', linewidth=2)
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    
    plt.tight_layout()
    plt.savefig('vit_training_metrics.png', dpi=300)
    print(" -> Saved 'vit_training_metrics.png'")

    def export_c_matrix(tensor_float, name):
        # Hardware QAT dictates scaling up by 64, rounding, and clamping
        int8_arr = torch.clamp(torch.round(tensor_float.detach() * 64.0), -128, 127).to(torch.int8)
        rows, cols = int8_arr.shape
        c_str = f"tensor16x16_t {name} = {{{{\n"
        for i in range(rows):
            row_vals = [f"{val.item():4d}" for val in int8_arr[i]]
            c_str += f"    {{ {', '.join(row_vals)} }}"
            c_str += ",\n" if i < rows - 1 else "\n"
        c_str += "}};\n\n"
        return c_str

    with open("vit_weights.c", "w") as f:
        f.write("// AUTOGENERATED INT8 WEIGHTS FOR CIMMETRY ViT-SWIGLU BLOCK\n\n")
        f.write("typedef struct { int8_t mat[16][16]; } tensor16x16_t;\n\n")
        
        f.write(export_c_matrix(model.W_q, "W_q"))
        f.write(export_c_matrix(model.W_k, "W_k"))
        f.write(export_c_matrix(model.W_v, "W_v"))
        f.write(export_c_matrix(model.W_o, "W_o"))
        f.write(export_c_matrix(model.W_up, "W_up"))
        f.write(export_c_matrix(model.W_gate, "W_gate"))
        f.write(export_c_matrix(model.W_down, "W_down"))
        
    print(" -> Saved 'vit_weights.c'")

    print("[*] Exporting INT8 .pth files for infer.py...")
    
    def get_int8_tensor(tensor_float):
        return torch.clamp(torch.round(tensor_float.detach() * 64.0), -128, 127).to(torch.int8)

    torch.save(get_int8_tensor(model.W_q), "W_q_int8.pth")
    torch.save(get_int8_tensor(model.W_k), "W_k_int8.pth")
    torch.save(get_int8_tensor(model.W_v), "W_v_int8.pth")
    torch.save(get_int8_tensor(model.W_o), "W_o_int8.pth")
    torch.save(get_int8_tensor(model.W_up), "W_up_int8.pth")
    torch.save(get_int8_tensor(model.W_gate), "W_gate_int8.pth")
    torch.save(get_int8_tensor(model.W_down), "W_down_int8.pth")
    
    print(" -> Saved all INT8 .pth files successfully.")

    print("[*] Exporting Test Images to C format...")

    def generate_c_struct(tensor, name, struct_type):
        rows, cols = tensor.shape
        c_str = f"{struct_type} {name} = {{\n    {{\n"
        for i in range(rows):
            row_vals = [f"{val.item():4d}" for val in tensor[i]]
            c_str += f"        {{ {', '.join(row_vals)} }}"
            c_str += ",\n" if i < rows - 1 else "\n"
        c_str += "    }}\n};\n\n"
        return c_str

    def format_image_for_hardware(img_8x8):
        # Slice 8x8 image into four 4x4 patches (16 elements each)
        patch_TL = img_8x8[0:4, 0:4].reshape(16)
        patch_TR = img_8x8[0:4, 4:8].reshape(16)
        patch_BL = img_8x8[4:8, 0:4].reshape(16)
        patch_BR = img_8x8[4:8, 4:8].reshape(16)
        return torch.stack([patch_TL, patch_TR, patch_BL, patch_BR], dim=0)

    # Grab some test images (cast to int8 to match hardware data types)
    test_steve_1 = steve_data[0].to(torch.int8)
    test_steve_2 = steve_data[1].to(torch.int8)
    test_alex_1 = alex_data[0].to(torch.int8)
    test_alex_2 = alex_data[1].to(torch.int8)

    with open("vit_test_images.c", "w") as f:
        f.write("// AUTOGENERATED INT8 TEST IMAGES (PRE-FORMATTED TO 4x16)\n\n")
        f.write("typedef struct { int8_t mat[4][16]; } tensor4x16_t;\n\n")
        
        img_s1_fmt = format_image_for_hardware(test_steve_1)
        f.write(generate_c_struct(img_s1_fmt, "test_steve_1", "tensor4x16_t"))
        
        img_s2_fmt = format_image_for_hardware(test_steve_2)
        f.write(generate_c_struct(img_s2_fmt, "test_steve_2", "tensor4x16_t"))
        
        img_a1_fmt = format_image_for_hardware(test_alex_1)
        f.write(generate_c_struct(img_a1_fmt, "test_alex_1", "tensor4x16_t"))
        
        img_a2_fmt = format_image_for_hardware(test_alex_2)
        f.write(generate_c_struct(img_a2_fmt, "test_alex_2", "tensor4x16_t"))

    print(" -> Saved 'test_images.c'. Hardware validation suite is ready!")