import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from model import Float32HardwareSurrogate32x32, ContrastiveLoss
from export import export_firmware

if __name__ == "__main__":
    print("[*] Loading 32x32 dataset...")
    data = torch.load("dataset/processed_data_32x32.pt", weights_only=False)
    train_pairs, train_targets = data['train_pairs'], data['train_targets']
    test_pairs, test_targets = data['test_pairs'], data['test_targets']
    
    # Load raw arrays for anchor generation and test image exporting
    train_img, train_lbl = data['train_img'], data['train_lbl']
    test_img, test_lbl = torch.FloatTensor(data['test_img']), torch.tensor(data['test_lbl'])

    model = Float32HardwareSurrogate32x32()
    criterion = ContrastiveLoss(margin=3.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.5)

    epochs, batch_size = 300, 64
    train_losses, val_accuracies = [], []
    final_optimal_threshold = 1.0

    print("[*] Starting QAT Training for 32x32 Network...")
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(train_pairs.size()[0])
        epoch_loss = 0
        
        for i in range(0, train_pairs.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            sig1, sig2 = model(train_pairs[indices, 0], train_pairs[indices, 1])
            loss = criterion(sig1, sig2, train_targets[indices].unsqueeze(1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        train_losses.append(epoch_loss / (train_pairs.size()[0] / batch_size))
        
        # --- VALIDATION ---
        model.eval()
        with torch.no_grad():
            sig1, sig2 = model(test_pairs[:, 0], test_pairs[:, 1])
            distances = F.pairwise_distance(sig1, sig2)
            
            best_acc, best_th = 0, 1.0
            for th in np.arange(0.1, 3.0, 0.1):
                acc = ((distances < th).float() == test_targets).sum().item() / len(test_targets)
                if acc > best_acc:
                    best_acc, best_th = acc, th
            
            val_accuracies.append(best_acc)
            final_optimal_threshold = best_th

        if (epoch+1) % 25 == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {train_losses[-1]:.4f} | Val Acc: {val_accuracies[-1]*100:.1f}% (Thresh: {best_th:.1f})")

    # --- PLOTTING ---
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, color='lightseagreen', linewidth=2)
    plt.title('Contrastive Loss')
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, color='lightseagreen', linewidth=2)
    plt.title('Validation Accuracy')
    plt.tight_layout()
    plt.savefig('fyp_training_metrics.png', dpi=300)
    
    # --- ANCHOR GENERATION ---
    print("\n[*] Calculating Authorized Personnel Anchors...")
    anchors = []
    model.eval()
    with torch.no_grad():
        # Only generating anchors for the first 3 "Authorized" people out of the 40 trained
        for i in range(3):
            person_imgs = train_img[train_lbl == i]
            person_imgs_tensor = torch.FloatTensor(person_imgs)
            
            signatures = model.forward_once(person_imgs_tensor)
            
            anchor = torch.mean(signatures, dim=0)
            anchors.append(anchor)

    # --- FIRMWARE EXPORT ---
    export_firmware(model, test_img, test_lbl, anchors, final_optimal_threshold)