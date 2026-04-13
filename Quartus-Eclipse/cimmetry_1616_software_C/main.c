/*
 * main.c
 *
 *  Created on: Apr 13, 2026
 *      Author: TYC
 */
#include <stdio.h>
#include <stdint.h>
#include "system.h"
#include "io.h"

// ==============================================================================
// HARDWARE BASE ADDRESS & REGISTERS
// ==============================================================================
#define CIM_BASE CIMMETRY_1616_0_BASE

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

// ==============================================================================
// TENSOR DATATYPES like torch.tensor()
// ==============================================================================
// Strict typing for our matrices.
typedef struct {
    int8_t mat[4][16];
} tensor4x16_t;

typedef struct {
    int8_t mat[16][4];
} tensor16x4_t;

typedef struct {
    int8_t mat[4][4];
} tensor4x4_t;

// ==============================================================================
// THE ACCELERATOR FUNCTION like torch.matmul()
// ==============================================================================
tensor4x4_t cimmetry(tensor4x16_t A, tensor16x4_t B)
{
    tensor4x4_t C_out;

    // 1. Configure Strides & Bases
    CIM_WRITE_REG(REG_BASE_A,   0);
    CIM_WRITE_REG(REG_BASE_B,   64);
    CIM_WRITE_REG(REG_BASE_C,   96);
    CIM_WRITE_REG(REG_STRIDE_A, 16);
    CIM_WRITE_REG(REG_STRIDE_B, 16);
    CIM_WRITE_REG(REG_STRIDE_C, 4);

    // 2. Load Matrix A into Scratchpad
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 16; j++) {
            // Access the inner array using .mat
            CIM_WRITE_SPAD(0 + (i * 16) + j, A.mat[i][j]);
        }
    }

    // 3. Load Matrix B (Hardware transpose translation)
    for (int col = 0; col < 4; col++) {
        for (int row = 0; row < 16; row++) {
            // Access the inner array using .mat
            CIM_WRITE_SPAD(64 + (col * 16) + row, B.mat[row][col]);
        }
    }

    // 4. Trigger Computation
    CIM_WRITE_REG(REG_CTRL, 0x01);

    // 5. Poll for Completion
    uint32_t status = 0;
    while ((status & 0x02) == 0) {
        status = CIM_READ_REG(REG_CTRL);
    }

    // 6. Read Results into our Output Tensor Struct
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            C_out.mat[i][j] = CIM_READ_SPAD(96 + (i * 4) + j);
        }
    }

    return C_out;
}

// ==============================================================================
// MAIN TEST FUNCTION
// ==============================================================================
int main()
{
    printf("==========================================\n");
    printf("   CIMmetry-1616 Strict Tensor Wrapper\n");
    printf("==========================================\n");

    // Initialize the A Tensor.
    // Notice the double braces {{ }}: The outer brace is for the struct,
    // the inner brace is for the 2D array.
    tensor4x16_t A_input = {{
        { 1,  2, -1,  0,   1,  2, -1,  0,   1,  2, -1,  0,   1,  2, -1,  0},
        { 0,  1,  2, -1,   0,  1,  2, -1,   0,  1,  2, -1,   0,  1,  2, -1},
        {-1,  0,  1,  2,  -1,  0,  1,  2,  -1,  0,  1,  2,  -1,  0,  1,  2},
        { 2, -1,  0,  1,   2, -1,  0,  1,   2, -1,  0,  1,   2, -1,  0,  1}
    }};

    // Initialize the B Tensor.
    tensor16x4_t B_input = {{
        {1,  1,  2,  0},
        {1, -1,  0,  1},
        {1,  1, -2,  0},
        {1, -1,  0,  1},
        {1,  1,  2,  0},
        {1, -1,  0,  1},
        {1,  1, -2,  0},
        {1, -1,  0,  1},
        {1,  1,  2,  0},
        {1, -1,  0,  1},
        {1,  1, -2,  0},
        {1, -1,  0,  1},
        {1,  1,  2,  0},
        {1, -1,  0,  1},
        {1,  1, -2,  0},
        {1, -1,  0,  1}
    }};

    printf("Executing C = cimmetry(A, B)...\n");

    // Pure, clean syntax.
    tensor4x4_t C = cimmetry(A_input, B_input);

    printf("---- RESULT MATRIX C (4x4) ----\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%4d ", C.mat[i][j]);
        }
        printf("\n");
    }

    printf("\nEXPECTED RESULT MATRIX C:\n");
    printf("   8   -8   16    8\n");
    printf("   8    8  -16    0\n");
    printf("   8   -8  -16    8\n");
    printf("   8    8   16    0\n");
    printf("==========================================\n");

    return 0;
}

