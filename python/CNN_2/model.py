import torch
import torch.nn as nn
import torch.nn.functional as F

class Float32HardwareSurrogate32x32(nn.Module):
    def __init__(self):
        super().__init__()
        # Initializations
        self.W_conv = nn.Parameter(torch.randn(16, 16) * 0.1) 
        self.W_linear = nn.Parameter(torch.randn(16, 8) * 0.1)
        self.hw_scale = 64.0 
        
    def sim_hw(self, tensor):
        """The ultimate hardware digital twin function (with STE)."""
        # 1. Scale float up to integer space
        t_scaled = tensor * self.hw_scale
        
        # 2. THE STE TRICK: Round the value, but trick PyTorch into 
        # acting like the round() function doesn't exist during backprop.
        t_rounded = t_scaled + (torch.round(t_scaled) - t_scaled).detach()
        
        # 3. Enforce 8-bit register limits (Simulate the SPAD bounds)
        t_clamped = torch.clamp(t_rounded, min=-128.0, max=127.0)
        
        # 4. Scale back to float so PyTorch math continues
        return t_clamped / self.hw_scale

    def forward_once(self, img_float):
        # Quantize weights to hardware steps BEFORE doing the math
        qw_conv = self.sim_hw(self.W_conv)
        qw_linear = self.sim_hw(self.W_linear)
        
        # 1. Extract 64 Patches
        img_reshaped = img_float.view(-1, 1, 32, 32)
        matrix_A_cycle1 = F.unfold(img_reshaped, kernel_size=4, stride=4).transpose(1, 2)
        
        # 2. Conv -> ReLU -> SPAD Clamp
        acc_cycle1 = torch.matmul(matrix_A_cycle1, qw_conv)
        acc_cycle1 = F.relu(acc_cycle1)
        conv_out = self.sim_hw(acc_cycle1)
        
        # 3. Quadrant Pooling
        grid = conv_out.view(-1, 8, 8, 16)
        quad_TL = torch.mean(grid[:, 0:4, 0:4, :], dim=(1, 2)) 
        quad_TR = torch.mean(grid[:, 0:4, 4:8, :], dim=(1, 2)) 
        quad_BL = torch.mean(grid[:, 4:8, 0:4, :], dim=(1, 2)) 
        quad_BR = torch.mean(grid[:, 4:8, 4:8, :], dim=(1, 2)) 

        matrix_A_cycle2 = torch.stack([quad_TL, quad_TR, quad_BL, quad_BR], dim=1) 

        # 4. Cycle 2 Projection -> SPAD Clamp
        acc_cycle2 = torch.matmul(matrix_A_cycle2, qw_linear)
        final_embedding = self.sim_hw(acc_cycle2)
        
        return final_embedding.view(-1, 32)

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, sig1, sig2, label):
        euclidean_distance = F.pairwise_distance(sig1, sig2, keepdim=True)
        return torch.mean((label) * torch.pow(euclidean_distance, 2) +
                          (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


