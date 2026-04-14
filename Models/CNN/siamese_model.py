import torch
import torch.nn as nn
import torch.nn.functional as F

import torch

class INT8Siamese:
    def __init__(self):
        
        #simulating weights that have already been quantized to range [-128, 127].
        self.W_conv_int8 = torch.randint(-128, 127, (16, 4), dtype=torch.int8) 
        self.W_linear_int8 = torch.randint(-128, 127, (16, 4), dtype=torch.int8)

        #to scale an int32 accumulator back to INT8, we use a bit-shift.
        self.shift_amount = 6 #>> 6 is mathematically equivalent to dividing by 64
        
    def forward_once(self, img_int8):
        
       #conv2s stride 4, kernel size 4 -> each patch is 4x4
        patch_TL = img_int8[0:4, 0:4].reshape(16)
        patch_TR = img_int8[0:4, 4:8].reshape(16)
        patch_BL = img_int8[4:8, 0:4].reshape(16)
        patch_BR = img_int8[4:8, 4:8].reshape(16)
        
        #[4, 16] int8
        matrix_A_cycle1 = torch.stack([patch_TL, patch_TR, patch_BL, patch_BR], dim=0)

        #cast to int32 before multiplying.
        acc_cycle1 = torch.matmul(matrix_A_cycle1.to(torch.int32), self.W_conv_int8.to(torch.int32))
        
        #relu
        acc_cycle1 = torch.max(acc_cycle1, torch.tensor(0, dtype=torch.int32))
        
        #bit-shift right to scale down, then clamp to 8-bit limits.
        scaled_out = acc_cycle1 >> self.shift_amount
        conv_out_int8 = torch.clamp(scaled_out, min=-128, max=127).to(torch.int8)
        
        #memeory arrangement for linear layer
        matrix_A_cycle2 = torch.zeros((4, 16), dtype=torch.int8)
        write_address_col = 0
        
        for quadrant in range(4):
            for channel in range(4):
                matrix_A_cycle2[0, write_address_col] = conv_out_int8[quadrant, channel]
                write_address_col += 1

        #Linear layer
        #casting to 32-bit for the accumulation phase
        acc_cycle2 = torch.matmul(matrix_A_cycle2.to(torch.int32), self.W_linear_int8.to(torch.int32))
        
        #shift and clamp the final 4-value embedding back to INT8
        final_embedding = torch.clamp(acc_cycle2 >> self.shift_amount, min=-128, max=127).to(torch.int8)
        
        return final_embedding[0, :]

    def forward(self, img1_int8, img2_int8):
        return self.forward_once(img1_int8), self.forward_once(img2_int8)

#for quantization-aware training, we can use the same forward logic but with float32 tensors that simulate the quantization effects.
class Float32HardwareSurrogate(nn.Module):
    def __init__(self):
        super().__init__()
        #initialize weights randomly, but keep them small
        self.W_conv = nn.Parameter(torch.randn(16, 4) * 0.1) 
        self.W_linear = nn.Parameter(torch.randn(16, 4) * 0.1)
        
        #bit-shift of >> 6 in hardware is mathematically division by 2^6 (64)
        self.hardware_scale = 64.0 
        
    def forward_once(self, img_float):
        #image extraction 
        patch_TL = img_float[0:4, 0:4].reshape(16)
        patch_TR = img_float[0:4, 4:8].reshape(16)
        patch_BL = img_float[4:8, 0:4].reshape(16)
        patch_BR = img_float[4:8, 4:8].reshape(16)
        
        matrix_A_cycle1 = torch.stack([patch_TL, patch_TR, patch_BL, patch_BR], dim=0)
        
        #Conv -> ReLU -> Scale -> Hardware Clamp
        acc_cycle1 = torch.matmul(matrix_A_cycle1, self.W_conv)
        acc_cycle1 = F.relu(acc_cycle1)
        
        scaled_out = acc_cycle1 / self.hardware_scale
        #8-bit register limits during training
        conv_out_simulated = torch.clamp(scaled_out, min=-128.0, max=127.0) 
        
        #memory arrangement
        matrix_A_cycle2 = torch.zeros((4, 16))
        write_address_col = 0
        for quadrant in range(4):
            for channel in range(4):
                matrix_A_cycle2[0, write_address_col] = conv_out_simulated[quadrant, channel]
                write_address_col += 1

        #Linear -> Scale -> Hardware Clamp
        acc_cycle2 = torch.matmul(matrix_A_cycle2, self.W_linear)
        final_embedding = torch.clamp(acc_cycle2 / self.hardware_scale, min=-128.0, max=127.0) #make sure all values are within int8 range
        
        return final_embedding[0, :]

    def forward(self, img1, img2):
        return self.forward_once(img1), self.forward_once(img2)