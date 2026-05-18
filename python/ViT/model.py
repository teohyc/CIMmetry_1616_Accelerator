import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HardwareViTSurrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.hw_scale = 64.0 
        
        self.W_q = nn.Parameter(torch.randn(16, 16) * 0.1)
        self.W_k = nn.Parameter(torch.randn(16, 16) * 0.1)
        self.W_v = nn.Parameter(torch.randn(16, 16) * 0.1)
        self.W_o = nn.Parameter(torch.randn(16, 16) * 0.1)
        
        self.W_up = nn.Parameter(torch.randn(16, 16) * 0.1)
        self.W_gate = nn.Parameter(torch.randn(16, 16) * 0.1)
        self.W_down = nn.Parameter(torch.randn(16, 16) * 0.1)

    def sim_hw(self, tensor):
        t_scaled = tensor * self.hw_scale
        t_rounded = t_scaled + (torch.round(t_scaled) - t_scaled).detach()
        t_clamped = torch.clamp(t_rounded, min=-128.0, max=127.0)
        return t_clamped / self.hw_scale

    def sim_layer_norm(self, tensor):
        mean = tensor.mean(dim=-1, keepdim=True)
        var = tensor.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (tensor - mean) / torch.sqrt(var + 1e-5)
        return self.sim_hw(normalized)

    def forward_once(self, img_float):
        B = img_float.size(0)
        patches = torch.stack([
            img_float[:, 0:4, 0:4].reshape(B, 16),
            img_float[:, 0:4, 4:8].reshape(B, 16),
            img_float[:, 4:8, 0:4].reshape(B, 16),
            img_float[:, 4:8, 4:8].reshape(B, 16) 
        ], dim=1)
        
        qw_q, qw_k, qw_v = self.sim_hw(self.W_q), self.sim_hw(self.W_k), self.sim_hw(self.W_v)
        qw_o = self.sim_hw(self.W_o)
        qw_up, qw_gate, qw_down = self.sim_hw(self.W_up), self.sim_hw(self.W_gate), self.sim_hw(self.W_down)

        # Self-Attention
        Q = self.sim_hw(torch.matmul(patches, qw_q))
        K = self.sim_hw(torch.matmul(patches, qw_k))
        V = self.sim_hw(torch.matmul(patches, qw_v))

        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(16)
        probs = F.softmax(scores, dim=-1)
        q_probs = self.sim_hw(probs)

        attn_out = self.sim_hw(torch.matmul(q_probs, V))
        attn_proj = self.sim_hw(torch.matmul(attn_out, qw_o))

        # Add & Norm
        res_1 = self.sim_hw(patches + attn_proj)
        norm_1 = self.sim_layer_norm(res_1)

        # SwiGLU FFN
        up_branch = self.sim_hw(torch.matmul(norm_1, qw_up))
        gate_branch = self.sim_hw(torch.matmul(norm_1, qw_gate))
        
        swish_activated = up_branch * torch.sigmoid(up_branch)
        ffn_activated = self.sim_hw(swish_activated * gate_branch)
        
        ffn_out = self.sim_hw(torch.matmul(ffn_activated, qw_down))
        final_tokens = self.sim_hw(norm_1 + ffn_out)

        # Mean Pool
        return torch.mean(final_tokens, dim=1)

    def forward(self, img1, img2):
        return self.forward_once(img1), self.forward_once(img2)
    
    
class INT8ViTSiamese:
    def __init__(self):
        self.shift_amount = 6
        self.W_q = None
        self.W_k = None
        self.W_v = None
        self.W_o = None
        self.W_up = None
        self.W_gate = None
        self.W_down = None

    def hardware_proj(self, A_int8, W_int8):
        acc = torch.matmul(A_int8.to(torch.int32), W_int8.to(torch.int32))
        return torch.clamp(acc >> self.shift_amount, min=-128, max=127).to(torch.int8)

    def layer_norm_software(self, tensor_int8):
        # Emulates the Nios II floating-point math layer
        t_float = tensor_int8.float()
        mean = t_float.mean(dim=-1, keepdim=True)
        var = t_float.var(dim=-1, keepdim=True, unbiased=False)
        norm = (t_float - mean) / torch.sqrt(var + 1e-5)
        quant = torch.clamp(torch.round(norm * 64.0), min=-128, max=127)
        return quant.to(torch.int8)

    def forward_once(self, img_int8):
        # 1. Patch Extraction
        patch_TL = img_int8[0:4, 0:4].reshape(16)
        patch_TR = img_int8[0:4, 4:8].reshape(16)
        patch_BL = img_int8[4:8, 0:4].reshape(16)
        patch_BR = img_int8[4:8, 4:8].reshape(16)
        patches = torch.stack([patch_TL, patch_TR, patch_BL, patch_BR], dim=0)

        # 2. Q, K, V Generation
        Q = self.hardware_proj(patches, self.W_q)
        K = self.hardware_proj(patches, self.W_k)
        V = self.hardware_proj(patches, self.W_v)

        # 3. Attention Scores
        raw_scores = torch.matmul(Q.to(torch.int32), K.transpose(0, 1).to(torch.int32))
        scaled_scores = raw_scores >> 2 # divide by 4 (sqrt(16))

        # 4. Nios II Softmax
        scores_float = scaled_scores.float()
        probs = F.softmax(scores_float, dim=-1)
        q_probs = torch.clamp(torch.round(probs * 64.0), min=-128, max=127).to(torch.int8)

        # 5. Attn Out & Proj
        attn_out = self.hardware_proj(q_probs, V)
        attn_proj = self.hardware_proj(attn_out, self.W_o)

        # 6. Residual & Norm
        res_1 = torch.clamp(patches.to(torch.int32) + attn_proj.to(torch.int32), -128, 127).to(torch.int8)
        norm_1 = self.layer_norm_software(res_1)

        # 7. SwiGLU FFN
        up_branch = self.hardware_proj(norm_1, self.W_up)
        gate_branch = self.hardware_proj(norm_1, self.W_gate)

        # PRISM-16 Emulator
        up_float = up_branch.float()
        swish = torch.round(up_float * torch.sigmoid(up_float)).to(torch.int32)
        ffn_activated = torch.clamp((swish * gate_branch.to(torch.int32)) >> self.shift_amount, -128, 127).to(torch.int8)

        # 8. Down Proj & Final Residual
        ffn_out = self.hardware_proj(ffn_activated, self.W_down)
        final_tokens = torch.clamp(norm_1.to(torch.int32) + ffn_out.to(torch.int32), -128, 127).to(torch.int8)

        # Return mean-pooled embedding
        return torch.mean(final_tokens.float(), dim=0)