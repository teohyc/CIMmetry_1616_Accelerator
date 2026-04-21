/*
 * main.c
 *
 * Created on: Apr 21, 2026
 * Author: TYC
 * Description: Dual-Core SoC Transformer SwiGLU Block Verification
 * (CIMmetry-1616 + PRISM-16)
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "system.h"
#include "io.h"

// ==============================================================================
// HARDWARE BASE ADDRESSES
// ==============================================================================
#define CIM_BASE CIMMETRY_1616_0_BASE
#define PRISM_BASE PRISM_16_0_BASE

#define REG_CTRL     0
#define REG_BASE_A   1
#define REG_BASE_B   2
#define REG_BASE_C   3
#define REG_STRIDE_A 4
#define REG_STRIDE_B 5
#define REG_STRIDE_C 6
#define SPAD_OFFSET  0x1000

#define CIM_WRITE_REG(offset, data) IOWR(CIM_BASE, offset, data)
#define CIM_READ_REG(offset)        IORD(CIM_BASE, offset)
#define CIM_WRITE_SPAD(addr, data)  IOWR(CIM_BASE, SPAD_OFFSET + (addr), data)
#define CIM_READ_SPAD(addr)         (int8_t)IORD(CIM_BASE, SPAD_OFFSET + (addr))

#define PRISM_OFFSET_X    0
#define PRISM_OFFSET_Y    16
#define PRISM_OFFSET_CTRL 32
#define PRISM_OFFSET_Z    48
#define PRISM_CMD_SWIGLU  ((3 << 1) | 1) // 0x07

typedef struct { int8_t mat[4][16]; } tensor4x16_t;
typedef struct { int8_t mat[16][4]; } tensor16x4_t;
typedef struct { int8_t mat[4][4]; }  tensor4x4_t;
typedef struct { int8_t mat[16][16];} tensor16x16_t;

// ==============================================================================
// THE GOLDEN VECTOR MATRICES (Crafted for 8-Bit Hardware Verification)
// ==============================================================================

// Input Matrix (Zero-Mean pairs to test Layer Norm perfectly)
tensor4x16_t X = {{
    { 10, -10,   6,  -6,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 }, // Tok 0
    { 16, -16,   8,  -8,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 }, // Tok 1
    { 24, -24,  12, -12,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 }, // Tok 2
    { 32, -32,  16, -16,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 }  // Tok 3
}};

// W_q & W_k: Shift Data Right by 3 Columns
tensor16x16_t W_q = {{
    {0,0,0,64,0,0,0,0,0,0,0,0,0,0,0,0}, {0,0,0,0,64,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,64,0,0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,64,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,64,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0,0,64,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,64,0,0,0,0}, {0,0,0,0,0,0,0,0,0,0,0,0,64,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,64,0,0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0,64,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,64}, {64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, {0,0,64,0,0,0,0,0,0,0,0,0,0,0,0,0}
}};
tensor16x16_t W_k = {0}; // Will mirror W_q in main()

// W_v: Shift Data Right by 5 Columns
tensor16x16_t W_v = {{
    {0,0,0,0,0,64,0,0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,64,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,64,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0,0,64,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,64,0,0,0,0}, {0,0,0,0,0,0,0,0,0,0,0,0,64,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,64,0,0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0,64,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,64}, {64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, {0,0,64,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,64,0,0,0,0,0,0,0,0,0,0,0,0}, {0,0,0,0,64,0,0,0,0,0,0,0,0,0,0,0}
}};

// W_o: Shift Data Right by 11 Columns (Realigns 5 + 11 = 16 = Index 0!)
tensor16x16_t W_o = {{
    {0,0,0,0,0,0,0,0,0,0,0,64,0,0,0,0}, {0,0,0,0,0,0,0,0,0,0,0,0,64,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,64,0,0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0,64,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,64}, {64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, {0,0,64,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,64,0,0,0,0,0,0,0,0,0,0,0,0}, {0,0,0,0,64,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,64,0,0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,64,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,64,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0,0,64,0,0,0,0,0}
}};

// W_up & W_gate: Shift Data Right by 4 Columns (Both MUST align for SwiGLU!)
tensor16x16_t W_up = {{
    {0,0,0,0,64,0,0,0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,64,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,64,0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0,64,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,64,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0,0,0,64,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,64,0,0,0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,64,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,64,0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,64},
    {64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, {0,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,64,0,0,0,0,0,0,0,0,0,0,0,0,0}, {0,0,0,64,0,0,0,0,0,0,0,0,0,0,0,0}
}};
tensor16x16_t W_gate = {0}; // Will mirror W_up in main()

// W_down: Shift Data Right by 12 Columns (Realigns 4 + 12 = 16 = Index 0!)
tensor16x16_t W_down = {{
    {0,0,0,0,0,0,0,0,0,0,0,0,64,0,0,0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,64,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,64,0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,64},
    {64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, {0,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,64,0,0,0,0,0,0,0,0,0,0,0,0,0}, {0,0,0,64,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,64,0,0,0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,64,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,64,0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0,64,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,64,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0,0,0,64,0,0,0,0}
}};

tensor4x16_t Q = {0}, K = {0}, V = {0};
tensor4x4_t raw_scores = {0};
tensor4x16_t probs_padded = {0}, attn_out = {0}, attn_proj = {0};
tensor4x16_t res_1 = {0}, norm_1 = {0};
tensor4x16_t up_branch = {0}, gate_branch = {0}, ffn_activated = {0}, final_out = {0};


void print_4x16(const char* label, tensor4x16_t* t) {
    printf("%s:\n", label);
    for(int r=0; r<4; r++){
        printf("  Tok %d: [ ", r);
        for(int c=0; c<16; c++) printf("%4d ", t->mat[r][c]);
        printf("]\n");
    }
    printf("\n");
}
void print_4x4(const char* label, tensor4x4_t* t) {
    printf("%s:\n", label);
    for(int r=0; r<4; r++){
        printf("  Tok %d: [ ", r);
        for(int c=0; c<4; c++) printf("%4d ", t->mat[r][c]);
        printf("]\n");
    }
    printf("\n");
}

void cimmetry_primitive(tensor4x16_t* A, tensor16x4_t* B, tensor4x4_t* C_out) {
    CIM_WRITE_REG(REG_BASE_A,   0);
    CIM_WRITE_REG(REG_BASE_B,   64);
    CIM_WRITE_REG(REG_BASE_C,   0);
    CIM_WRITE_REG(REG_STRIDE_A, 16);
    CIM_WRITE_REG(REG_STRIDE_B, 16);
    CIM_WRITE_REG(REG_STRIDE_C, 4);
    for (int i = 0; i < 4; i++) for (int j = 0; j < 16; j++) CIM_WRITE_SPAD(0 + (i * 16) + j, A->mat[i][j]);
    for (int col = 0; col < 4; col++) for (int row = 0; row < 16; row++) CIM_WRITE_SPAD(64 + (col * 16) + row, B->mat[row][col]);
    CIM_WRITE_REG(REG_CTRL, 0x01);
    while ((CIM_READ_REG(REG_CTRL) & 0x02) == 0);
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) C_out->mat[i][j] = CIM_READ_SPAD(0 + (i * 4) + j);
}

void cimmetry_proj_16x16(tensor4x16_t* A, tensor16x16_t* W, tensor4x16_t* output) {
    tensor16x4_t W_slice;
    tensor4x4_t tile_out;
    for (int tile = 0; tile < 4; tile++) {
        for (int row = 0; row < 16; row++) for (int col = 0; col < 4; col++) W_slice.mat[row][col] = W->mat[row][tile * 4 + col];
        cimmetry_primitive(A, &W_slice, &tile_out);
        for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) output->mat[r][tile * 4 + c] = tile_out.mat[r][c];
    }
}

void cimmetry_attention_scores(tensor4x16_t* Q, tensor4x16_t* K, tensor4x4_t* scores) {
    tensor16x4_t K_transposed;
    for(int r=0; r<4; r++) for(int c=0; c<16; c++) K_transposed.mat[c][r] = K->mat[r][c];
    cimmetry_primitive(Q, &K_transposed, scores);
}

void run_prism_swiglu(tensor4x16_t* X_in, tensor4x16_t* Y_in, tensor4x16_t* Z_out) {
    for (int row = 0; row < 4; row++) {
        uint32_t* X_word = (uint32_t*)&X_in->mat[row][0];
        uint32_t* Y_word = (uint32_t*)&Y_in->mat[row][0];
        uint32_t* Z_word = (uint32_t*)&Z_out->mat[row][0];
        for (int i = 0; i < 4; i++) {
            IOWR_32DIRECT(PRISM_BASE, PRISM_OFFSET_X + (i * 4), X_word[i]);
            IOWR_32DIRECT(PRISM_BASE, PRISM_OFFSET_Y + (i * 4), Y_word[i]);
        }
        IOWR_32DIRECT(PRISM_BASE, PRISM_OFFSET_CTRL, PRISM_CMD_SWIGLU);
        while ((IORD_32DIRECT(PRISM_BASE, PRISM_OFFSET_CTRL) & 0x02) == 0);
        for (int i = 0; i < 4; i++) Z_word[i] = IORD_32DIRECT(PRISM_BASE, PRISM_OFFSET_Z + (i * 4));
    }
}

void software_softmax_real(tensor4x4_t* scores, tensor4x16_t* probs_padded) {
    memset(probs_padded, 0, sizeof(tensor4x16_t));
    for(int r = 0; r < 4; r++) {
        float max_val = -9999.0f;
        for(int c = 0; c < 4; c++) if((float)scores->mat[r][c] > max_val) max_val = (float)scores->mat[r][c];
        float sum = 0.0f;
        float exps[4];
        for(int c = 0; c < 4; c++) {
            exps[c] = expf((float)scores->mat[r][c] - max_val);
            sum += exps[c];
        }
        for(int c = 0; c < 4; c++) {
            float prob = exps[c] / sum;
            probs_padded->mat[r][c] = (int8_t)(prob * 64.0f);
        }
    }
}

void software_residual(tensor4x16_t* A, tensor4x16_t* B, tensor4x16_t* C) {
    for(int r = 0; r < 4; r++) {
        for(int c = 0; c < 16; c++) {
            int16_t sum = A->mat[r][c] + B->mat[r][c];
            C->mat[r][c] = (sum > 127) ? 127 : ((sum < -128) ? -128 : sum);
        }
    }
}

void software_layer_norm(tensor4x16_t* input, tensor4x16_t* output) {
    float epsilon = 1e-5f;
    for(int r = 0; r < 4; r++) {
        int32_t sum = 0;
        for(int c = 0; c < 16; c++) sum += input->mat[r][c];
        float mean = (float)sum / 16.0f;
        float variance_sum = 0.0f;
        for(int c = 0; c < 16; c++) {
            float diff = (float)input->mat[r][c] - mean;
            variance_sum += (diff * diff);
        }
        float stdev = sqrtf((variance_sum / 16.0f) + epsilon);
        for(int c = 0; c < 16; c++) {
            float normalized = ((float)input->mat[r][c] - mean) / stdev;
            int16_t quant = (int16_t)(normalized * 16.0f);
            output->mat[r][c] = (quant > 127) ? 127 : ((quant < -128) ? -128 : quant);
        }
    }
}

// ==============================================================================
// MAIN EXECUTION PIPELINE
// ==============================================================================
int main() {
    printf("==========================================================================\n");
    printf("  Dual-Core SoC | Tranformer Block with SwiGLU \n");
    printf("==========================================================================\n\n");

    // Copy identical matrices to set up the golden vectors
    memcpy(&W_k, &W_q, sizeof(tensor16x16_t));
    memcpy(&W_gate, &W_up, sizeof(tensor16x16_t));

    print_4x16("[01] RAW INPUT TENSOR (X)", &X);

    // --------------------------------------------------------------------------
    // [MATH: Q, K, V Generation]
    // Equation: Q = X * W_q | K = X * W_k | V = X * W_v
    // Hardware: CIMmetry-1616 Matrix-Vector Multiplier Array
    // --------------------------------------------------------------------------
    cimmetry_proj_16x16(&X, &W_q, &Q);
    cimmetry_proj_16x16(&X, &W_k, &K);
    cimmetry_proj_16x16(&X, &W_v, &V);

    // --------------------------------------------------------------------------
    // [MATH: Attention Scoring]
    // Equation: Scores = Q * K^T
    // Hardware: CIMmetry-1616 Token-to-Token Attention Matrix
    // --------------------------------------------------------------------------
    cimmetry_attention_scores(&Q, &K, &raw_scores);

    // --------------------------------------------------------------------------
    // [MATH: Attention Scaling]
    // Equation: Scaled_Scores = Scores / sqrt(d_k)
    // Software: Nios II right bit-shift (>> 2 simulates division by 4)
    // --------------------------------------------------------------------------
    for(int r=0; r<4; r++)
        for(int c=0; c<4; c++) raw_scores.mat[r][c] = raw_scores.mat[r][c] >> 2;

    // --------------------------------------------------------------------------
    // [MATH: Softmax Probabilities]
    // Equation: Softmax(x_i) = exp(x_i) / sum(exp(x_j))
    // Software: Nios II Floating-Point Math
    // --------------------------------------------------------------------------
    software_softmax_real(&raw_scores, &probs_padded);

    // --------------------------------------------------------------------------
    // [MATH: Value Aggregation]
    // Equation: Attention_Output = Softmax * V
    // Hardware: CIMmetry-1616 Output Projection
    // --------------------------------------------------------------------------
    for(int tile=0; tile<4; tile++){
        tensor16x4_t V_padded_slice = {0};
        tensor4x4_t out_slice;
        for(int r=0; r<4; r++) for(int c=0; c<4; c++) V_padded_slice.mat[r][c] = V.mat[r][tile*4+c];
        cimmetry_primitive(&probs_padded, &V_padded_slice, &out_slice);
        for(int r=0; r<4; r++) for(int c=0; c<4; c++) attn_out.mat[r][tile*4+c] = out_slice.mat[r][c];
    }
    print_4x16("[02] ATTENTION HEAD OUTPUT (V Matrix)", &attn_out);

    // --------------------------------------------------------------------------
    // [MATH: Output Projection]
    // Equation: Attn_Proj = Attention_Output * W_o
    // Hardware: CIMmetry-1616 (Realigns permutations)
    // --------------------------------------------------------------------------
    cimmetry_proj_16x16(&attn_out, &W_o, &attn_proj);
    print_4x16("[03] W_o PROJECTION ", &attn_proj);

    // --------------------------------------------------------------------------
    // [MATH: Add & Norm (Residual 1)]
    // Equation: Norm_Out = LayerNorm(X + Attn_Proj)
    // Software: Nios II Saturation Addition & RMS/Layer Normalization
    // --------------------------------------------------------------------------
    software_residual(&X, &attn_proj, &res_1);
    software_layer_norm(&res_1, &norm_1);
    print_4x16("[04] NORMALIZED TENSOR", &norm_1);

    // --------------------------------------------------------------------------
    // [MATH: SwiGLU Branch Projections]
    // Equation: Up = Norm_Out * W_up | Gate = Norm_Out * W_gate
    // Hardware: CIMmetry-1616 routing into FFN branches
    // --------------------------------------------------------------------------
    cimmetry_proj_16x16(&norm_1, &W_up, &up_branch);
    cimmetry_proj_16x16(&norm_1, &W_gate, &gate_branch);

    // --------------------------------------------------------------------------
    // [MATH: SwiGLU Activation Function]
    // Equation: Activated = (Up * Sigmoid(Up)) * Gate
    // Hardware: PRISM-16 Dual-Core execution (LUT + DSP Multiplier)
    // --------------------------------------------------------------------------
    run_prism_swiglu(&up_branch, &gate_branch, &ffn_activated);
    print_4x16("[05] PRISM-16 SWIGLU OUTPUT (Negative Numbers Dead)", &ffn_activated);

    // --------------------------------------------------------------------------
    // [MATH: Final Down Projection]
    // Equation: Final_Out = Activated * W_down
    // Hardware: CIMmetry-1616 (Final realignment)
    // --------------------------------------------------------------------------
    cimmetry_proj_16x16(&ffn_activated, &W_down, &final_out);
    print_4x16("[06] FINAL SWIGLU FFN OUTPUT (Realigned to X!)", &final_out);

    return 0;
}

