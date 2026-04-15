import torch
import torch.nn as nn
import math

class FlowDiT(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, depth=2):
        super().__init__()

        #each path 4x4
        self.patch_dim = 16

        #patch embedding
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)

        #positional embeddings for 4 patches
        self.pos_embed = nn.Parameter(torch.randn(1, 4, embed_dim))

        #time embedding (sinusoidal + mlp)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        #transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim*2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        #output projection
        self.head = nn.Linear(embed_dim, self.patch_dim)

    def forward(self, x, t):
        #image: [B, 1, 8, 8]
        #time: [B, 1]
        B = x.shape[0]

        #split into 4 patches of 4x4
        p_TL = x[:, 0, 0:4, 0:4].reshape(B, 16)
        p_TR = x[:, 0, 0:4, 4:8].reshape(B, 16)
        p_BL = x[:, 0, 4:8, 0:4].reshape(B, 16)
        p_BR = x[:, 0, 4:8, 4:8].reshape(B, 16)

        #stack patches: [B, 4, 16]
        tokens = torch.stack([p_TL, p_TR, p_BL, p_BR], dim=1)

        #embed_patches and add position
        x_emb = self.patch_embed(tokens) + self.pos_embed

        #embed time and add
        t_emb = self.time_mlp(t).unsqueeze(1)
        x_emb = x_emb + t_emb

        #pass through transformer
        out_embed = self.transformer(x_emb)

        #predict velocity for each patch
        v_pred_tokens = self.head(out_embed)

        #reconstruct velocity field
        v_pred_img = torch.zeros(B, 1, 8, 8, device=x.device)
        v_pred_img[:, 0, 0:4, 0:4] = v_pred_tokens[:, 0, :].view(B, 4, 4)
        v_pred_img[:, 0, 0:4, 4:8] = v_pred_tokens[:, 1, :].view(B, 4, 4)
        v_pred_img[:, 0, 4:8, 0:4] = v_pred_tokens[:, 2, :].view(B, 4, 4)
        v_pred_img[:, 0, 4:8, 4:8] = v_pred_tokens[:, 3, :].view(B, 4, 4)

        return v_pred_img