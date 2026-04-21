import numpy as np
import math


def print_4x16(label, t):
    print(f"{label}:")
    for r in range(4):
        # Format exactly like the C console output
        row_str = " ".join([f"{x:4d}" for x in t[r]])
        print(f"  Tok {r}: [ {row_str} ]")
    print()

def print_4x4(label, t):
    print(f"{label}:")
    for r in range(4):
        row_str = " ".join([f"{x:4d}" for x in t[r]])
        print(f"  Tok {r}: [ {row_str} ]")
    print()

def gen_permutation_weight(offset_shift):
    """Generates the Golden Vector hardware-safe weight matrices."""
    W = np.zeros((16, 16), dtype=np.int32)
    for i in range(16):
        j = (i + offset_shift) % 16
        W[i, j] = 64
    return W

def cimmetry_hardware_proj(A, W):
    """
    Emulates the CIMmetry-1616 MAC array and SRAM write-back.
    Multiplies the matrices and applies the hardware >> 6 right-shift.
    """
    return (A @ W) // 64


print("==========================================================================")
print("Golden Vector Verification")
print("==========================================================================\n")

X = np.array([
    [ 10, -10,   6,  -6,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 ], # Tok 0
    [ 16, -16,   8,  -8,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 ], # Tok 1
    [ 24, -24,  12, -12,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 ], # Tok 2
    [ 32, -32,  16, -16,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 ]  # Tok 3
], dtype=np.int32)

W_q = gen_permutation_weight(3)
W_k = gen_permutation_weight(3)
W_v = gen_permutation_weight(5)
W_o = gen_permutation_weight(11)
W_up = gen_permutation_weight(4)
W_gate = gen_permutation_weight(4)
W_down = gen_permutation_weight(12)

print_4x16("[01] RAW INPUT TENSOR (X)", X)

# ==============================================================================
# PHASE 1 & 2: QKV & Attention
# ==============================================================================
Q = cimmetry_hardware_proj(X, W_q)
K = cimmetry_hardware_proj(X, W_k)
V = cimmetry_hardware_proj(X, W_v)

raw_scores = cimmetry_hardware_proj(Q, K.T)
scaled_scores = raw_scores >> 2

# Emulate Nios II Software Softmax
probs_padded = np.zeros((4, 16), dtype=np.int32)
for r in range(4):
    row = scaled_scores[r]
    max_val = np.max(row)
    exps = np.exp(row - max_val)
    sum_exps = np.sum(exps)
    for c in range(4):
        probs_padded[r, c] = int((exps[c] / sum_exps) * 64.0)

# THE FIX: Pad V to match the hardware's 16-element MAC array requirements
V_padded = np.zeros((16, 16), dtype=np.int32)
V_padded[:4, :] = V  

# Now the dimensions perfectly match: (4x16) @ (16x16)
full_attn_out = cimmetry_hardware_proj(probs_padded, V_padded)
print_4x16("[02] ATTENTION HEAD OUTPUT (V Matrix Shifted)", full_attn_out)

# ==============================================================================
# PHASE 3: Output Projection & Residual
# ==============================================================================
attn_proj = cimmetry_hardware_proj(full_attn_out, W_o)
print_4x16("[03] W_o PROJECTION (Realigned to X!)", attn_proj)

# Residual Clamp
res_1 = np.clip(X + attn_proj, -128, 127)

# Emulate Nios II Layer Norm
norm_1 = np.zeros((4, 16), dtype=np.int32)
for r in range(4):
    mean = np.mean(res_1[r])
    variance = np.mean((res_1[r] - mean)**2)
    stdev = math.sqrt(variance + 1e-5)
    for c in range(16):
        normalized = (res_1[r, c] - mean) / stdev
        quantized = int(normalized * 16.0)
        norm_1[r, c] = np.clip(quantized, -128, 127)
print_4x16("[04] NORMALIZED TENSOR", norm_1)

# ==============================================================================
# PHASE 4: SwiGLU FFN & Down Projection
# ==============================================================================
up_branch = cimmetry_hardware_proj(norm_1, W_up)
gate_branch = cimmetry_hardware_proj(norm_1, W_gate)

# Emulate PRISM-16 Core Execution
ffn_activated = np.zeros((4, 16), dtype=np.int32)
for r in range(4):
    for c in range(16):
        u = up_branch[r, c]
        # Prevent math domain error for very large negative numbers in Python
        sigmoid = 1.0 / (1.0 + math.exp(-u)) if u > -20 else 0.0
        swish_val = int(u * sigmoid) # Emulate integer LUT return
        ffn_activated[r, c] = (swish_val * gate_branch[r, c]) // 64
print_4x16("[05] PRISM-16 SWIGLU OUTPUT (Negative Numbers Dead)", ffn_activated)

final_out = cimmetry_hardware_proj(ffn_activated, W_down)
print_4x16("[06] FINAL FFN OUTPUT", final_out)