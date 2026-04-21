/*
 * main.c
 *
 *  Created on: Apr 21, 2026
 *      Author: TYC
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

// ==============================================================================
// TENSOR DATATYPES
// ==============================================================================
typedef struct { int8_t mat[4][16]; } tensor4x16_t;
typedef struct { int8_t mat[16][4]; } tensor16x4_t;
typedef struct { int8_t mat[4][4]; }  tensor4x4_t;
typedef struct { int8_t mat[16][16];} tensor16x16_t;

// ==============================================================================
// GLOBAL MEMORY (Prevents Nios II Stack Overflow)
// ==============================================================================
tensor4x16_t X = {0};
tensor16x16_t W_identity = {0};
tensor4x16_t Q = {0}, K = {0}, V = {0};
tensor4x4_t raw_scores = {0};
tensor4x16_t probs_padded = {0}, attn_out = {0}, res_1 = {0};
tensor4x16_t swish_branch = {0}, gate_branch = {0}, ffn_activated = {0};


// Helper function to print intermediate matrices
void print_4x16(const char* label, tensor4x16_t* t) {
    printf("%s:\n", label);
    for(int r=0; r<4; r++){
        printf("  Tok %d: [ ", r);
        for(int c=0; c<16; c++) printf("%3d ", t->mat[r][c]);
        printf("]\n");
    }
    printf("\n");
}

void print_4x4(const char* label, tensor4x4_t* t) {
    printf("%s:\n", label);
    for(int r=0; r<4; r++){
        printf("  Tok %d: [ ", r);
        for(int c=0; c<4; c++) printf("%3d ", t->mat[r][c]);
        printf("]\n");
    }
    printf("\n");
}

// ==============================================================================
// CIMMETRY TILE ORCHESTRATORS (CORE 0)
// ==============================================================================
void cimmetry_primitive(tensor4x16_t* A, tensor16x4_t* B, tensor4x4_t* C_out) {
    CIM_WRITE_REG(REG_BASE_A,   0);
    CIM_WRITE_REG(REG_BASE_B,   64);
    CIM_WRITE_REG(REG_BASE_C,   0);
    CIM_WRITE_REG(REG_STRIDE_A, 16);
    CIM_WRITE_REG(REG_STRIDE_B, 16);
    CIM_WRITE_REG(REG_STRIDE_C, 4);

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 16; j++)
            CIM_WRITE_SPAD(0 + (i * 16) + j, A->mat[i][j]);

    for (int col = 0; col < 4; col++)
        for (int row = 0; row < 16; row++)
            CIM_WRITE_SPAD(64 + (col * 16) + row, B->mat[row][col]);

    CIM_WRITE_REG(REG_CTRL, 0x01);
    while ((CIM_READ_REG(REG_CTRL) & 0x02) == 0);

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            C_out->mat[i][j] = CIM_READ_SPAD(0 + (i * 4) + j);
}

void cimmetry_proj_16x16(tensor4x16_t* A, tensor16x16_t* W, tensor4x16_t* output) {
    tensor16x4_t W_slice;
    tensor4x4_t tile_out;

    for (int tile = 0; tile < 4; tile++) {
        for (int row = 0; row < 16; row++) {
            for (int col = 0; col < 4; col++) {
                W_slice.mat[row][col] = W->mat[row][tile * 4 + col];
            }
        }
        cimmetry_primitive(A, &W_slice, &tile_out);
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                output->mat[r][tile * 4 + c] = tile_out.mat[r][c];
            }
        }
    }
}

void cimmetry_attention_scores(tensor4x16_t* Q, tensor4x16_t* K, tensor4x4_t* scores) {
    tensor16x4_t K_transposed;
    for(int r=0; r<4; r++){
        for(int c=0; c<16; c++){
            K_transposed.mat[c][r] = K->mat[r][c];
        }
    }
    cimmetry_primitive(Q, &K_transposed, scores);
}

// ==============================================================================
// PRISM-16 SWIGLU EXECUTION (CORE 1)
// ==============================================================================
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

        for (int i = 0; i < 4; i++) {
            Z_word[i] = IORD_32DIRECT(PRISM_BASE, PRISM_OFFSET_Z + (i * 4));
        }
    }
}

// ==============================================================================
// NIOS II GLUE LOGIC
// ==============================================================================
void software_softmax_mock(tensor4x4_t* scores, tensor4x16_t* probs_padded) {
    memset(probs_padded, 0, sizeof(tensor4x16_t));
    for(int r = 0; r < 4; r++) {
        int max_idx = 0;
        int8_t max_val = -128;
        for(int c = 0; c < 4; c++) {
            if(scores->mat[r][c] > max_val) {
                max_val = scores->mat[r][c];
                max_idx = c;
            }
        }
        // SCALED IDENTITY FIX: Set to 64 so it survives the V-matrix projection quantization!
        probs_padded->mat[r][max_idx] = 64;
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

// ==============================================================================
// MAIN TRANSFORMER DIAGNOSTICS
// ==============================================================================
int main() {
    printf("==========================================================================\n");
    printf("  Dual-Core SoC | Transformer SwiGLU Block Diagnostics (Quantization Safe)\n");
    printf("==========================================================================\n\n");

    // 1. QUANTIZATION SAFE DATA FIX:

    for(int r=0; r<4; r++) {
        X.mat[r][0] = (r + 1) * 16; // Row 0=[16], Row 1=[32], Row 2=[48], Row 3=[64]
    }

    // Set diagonal to 64 (Acts as 1.0 in a system that divides by 64)
    for(int i=0; i<16; i++) W_identity.mat[i][i] = 64;

    print_4x16("[01] INPUT Sparse Tensor X", &X);

    // ---------------------------------------------------------
    // PHASE 1 & 2: QKV & Attention
    // ---------------------------------------------------------
    printf("\n[*] Core 0: Generating Q, K, V Projections...\n");
    cimmetry_proj_16x16(&X, &W_identity, &Q);
    print_4x16("[02] Q Matrix Output", &Q);

    cimmetry_proj_16x16(&X, &W_identity, &K);
    print_4x16("[03] K Matrix Output", &K);

    cimmetry_proj_16x16(&X, &W_identity, &V);
    print_4x16("[04] V Matrix Output", &V);

    printf("\n[*] Core 0: Computing Q * K^T Attention Scores...\n");
    cimmetry_attention_scores(&Q, &K, &raw_scores);
    print_4x4("[05] Raw Attention Scores (Before Scale)", &raw_scores);

    // Scale down before softmax
    for(int r=0; r<4; r++) {
        for(int c=0; c<4; c++) {
            raw_scores.mat[r][c] = raw_scores.mat[r][c] >> 2;
        }
    }
    print_4x4("[06] Scaled Attention Scores (>> 2)", &raw_scores);

    printf("\n[*] Nios II: Applying Softmax...\n");
    software_softmax_mock(&raw_scores, &probs_padded);
    print_4x16("[07] Softmax Probabilities (Zero-Padded, Max=64)", &probs_padded);

    printf("\n[*] Core 0: Computing Softmax * V...\n");
    for(int tile=0; tile<4; tile++){
        tensor16x4_t V_padded_slice = {0};
        tensor4x4_t out_slice;
        for(int r=0; r<4; r++){
            for(int c=0; c<4; c++) V_padded_slice.mat[r][c] = V.mat[r][tile*4+c];
        }
        cimmetry_primitive(&probs_padded, &V_padded_slice, &out_slice);
        for(int r=0; r<4; r++){
            for(int c=0; c<4; c++) attn_out.mat[r][tile*4+c] = out_slice.mat[r][c];
        }
    }
    print_4x16("[08] Attention Output (Softmax * V)", &attn_out);

    // ---------------------------------------------------------
    // PHASE 3 & 4: Residuals & PRISM-16 SwiGLU
    // ---------------------------------------------------------
    printf("\n[*] Nios II: Applying First Residual Connection...\n");
    software_residual(&X, &attn_out, &res_1);
    print_4x16("[09] Residual 1 (X + Attn_Out)", &res_1);

    printf("\n[*] Core 0: Projecting Swish and Gate Branches...\n");
    cimmetry_proj_16x16(&res_1, &W_identity, &swish_branch);
    print_4x16("[10] Swish Branch Input (Core 0 Output)", &swish_branch);

    // Cancel the bit-shift for exact math checking
    for(int r=0; r<4; r++) {
        for(int c=0; c<16; c++) gate_branch.mat[r][c] = 64;
    }
    print_4x16("[11] Gate Branch Input (Identity Multiplier 64)", &gate_branch);

    printf("\n[*] Core 1 (PRISM-16): Executing Vector SwiGLU Operation...\n");
    run_prism_swiglu(&swish_branch, &gate_branch, &ffn_activated);

    print_4x16("[12] FINAL PRISM-16 SwiGLU OUTPUT", &ffn_activated);

    printf("==========================================================================\n");
    printf("  EXPECTED SWISH VERIFICATION (Col 0 Only, Scale=16.0):\n");
    printf("  Token 0 (Input 16) -> Will scale against Swish Curve\n");
    printf("  Token 1 (Input 32) -> Will scale against Swish Curve\n");
    printf("  Token 2 (Input 48) -> Will scale against Swish Curve\n");
    printf("  Token 3 (Input 64) -> Will scale against Swish Curve\n");
    printf("==========================================================================\n");

    return 0;
}
