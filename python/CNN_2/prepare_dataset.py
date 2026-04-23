import torch
import numpy as np
import cv2
import itertools
import random
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
import os

def save_dataset_preview(images_64, images_32, filename="dataset_preview_32x32.png"):
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    fig.suptitle("Olivetti 40-Class Dataset: 64x64 vs 32x32", fontsize=12)
    for i in range(5):
        idx = i * 10 
        axes[0, i].imshow(images_64[idx], cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Person {i}")
        
        axes[1, i].imshow(images_32[idx], cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title(f"32x32")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

def create_balanced_pairs(images, labels):
    genuine_pairs, imposter_pairs = [], []
    num_classes = len(np.unique(labels))
    class_indices = [np.where(labels == i)[0] for i in np.unique(labels)]
    
    # 1. Generate ALL 100% Unique Genuine Pairs
    for cls in range(num_classes):
        for idx1, idx2 in itertools.combinations(class_indices[cls], 2):
            genuine_pairs.append([images[idx1], images[idx2]])

    # 2. Randomly sample Imposter Pairs to perfectly match the Genuine count (50/50 Split)
    target_imposters = len(genuine_pairs)
    while len(imposter_pairs) < target_imposters:
        cls1, cls2 = random.sample(range(num_classes), 2)
        idx1 = random.choice(class_indices[cls1])
        idx2 = random.choice(class_indices[cls2])
        imposter_pairs.append([images[idx1], images[idx2]])

    # Combine and Shuffle
    pairs = genuine_pairs + imposter_pairs
    targets = [1] * len(genuine_pairs) + [0] * len(imposter_pairs)
    
    pairs_tensor = torch.FloatTensor(np.array(pairs))
    targets_tensor = torch.FloatTensor(np.array(targets))
    
    indices = torch.randperm(len(pairs))
    return pairs_tensor[indices], targets_tensor[indices]

if __name__ == "__main__":
    os.makedirs("dataset", exist_ok=True)
    print("[*] Downloading Full 40-Person Olivetti Dataset...")
    
    data = fetch_olivetti_faces(data_home="./dataset", shuffle=False)
    images_64 = data.images
    labels = data.target

    images_32 = np.array([cv2.resize(img, (32, 32)) for img in images_64])
    save_dataset_preview(images_64, images_32, filename="dataset/dataset_preview_32x32.png")
    images_32 = (images_32 - 0.5) * 2.0

    # SPLIT: 8 photos for training, 2 for testing (for ALL 40 people)
    train_img, train_lbl, test_img, test_lbl = [], [], [], []
    for i in range(40):
        person_imgs = images_32[labels == i]
        train_img.extend(person_imgs[:8])
        train_lbl.extend([i] * 8)
        test_img.extend(person_imgs[8:])
        test_lbl.extend([i] * 2)

    train_img, train_lbl = np.array(train_img), np.array(train_lbl)
    test_img, test_lbl = np.array(test_img), np.array(test_lbl)

    print("[*] Generating Perfectly Balanced 50/50 Pairs...")
    train_pairs, train_targets = create_balanced_pairs(train_img, train_lbl)
    test_pairs, test_targets = create_balanced_pairs(test_img, test_lbl)
    
    print(f"[*] Training Pairs: {len(train_pairs)} | Testing Pairs: {len(test_pairs)}")

    save_path = "dataset/processed_data_32x32.pt"
    torch.save({
        'train_pairs': train_pairs, 'train_targets': train_targets,
        'test_pairs': test_pairs, 'test_targets': test_targets,
        'train_img': train_img, 'train_lbl': train_lbl,
        'test_img': test_img, 'test_lbl': test_lbl
    }, save_path)
    print(f"[*] Success! Full 40-Class Dataset saved.")