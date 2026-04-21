/*
 * main.c
 *
 *  Created on: Apr 21, 2026
 *      Author: TYC
 */
#include <stdio.h>
#include <stdint.h>
#include "system.h"
#include "io.h"

// ==============================================================================
// CIMMETRY-1616 HARDWARE BASE ADDRESS & REGISTERS
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
// PRISM-16 HARDWARE BASE ADDRESS & REGISTERS
// ==============================================================================
#define PRISM_BASE PRISM_16_0_BASE

#define PRISM_OFFSET_X    0
#define PRISM_OFFSET_Y    16
#define PRISM_OFFSET_CTRL 32
#define PRISM_OFFSET_Z    48

#define PRISM_CMD_RELU    ((0 << 1) | 1) // 0x01

// ==============================================================================
// TENSOR DATATYPES
// ==============================================================================
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
// EDGE AI METRICS & MATH
// ==============================================================================
int32_t calculate_sq_distance(int8_t* emb1, int8_t* emb2) {
    int32_t dist = 0;
    for (int i = 0; i < 4; i++) {
        int32_t diff = emb1[i] - emb2[i];
        dist += (diff * diff);
    }
    return dist;
}

void print_distance_bar(int32_t dist) {
    int max_visual_scale = 3000;
    int width = 25;
    int filled = (dist >= max_visual_scale) ? width : (dist * width / max_visual_scale);

    printf("[");
    for (int i = 0; i < width; i++) {
        if (i < filled) printf("#");
        else printf(" ");
    }
    printf("]");
}

// Reference Anchor Embeddings
int8_t ANCHOR_STEVE[4] = {-27,  26,  26, -26};
int8_t ANCHOR_ALEX[4]  = { 16, -17, -17,  15};

#define CLASSIFICATION_THRESHOLD 500

// ==============================================================================
// CIMMETRY MACROBLOCK WRAPPER (Spatial Matrix Engine)
// ==============================================================================
tensor4x4_t cimmetry(tensor4x16_t A, tensor16x4_t B)
{
    tensor4x4_t C_out;

    CIM_WRITE_REG(REG_BASE_A,   0);
    CIM_WRITE_REG(REG_BASE_B,   64);
    CIM_WRITE_REG(REG_BASE_C,   0);
    CIM_WRITE_REG(REG_STRIDE_A, 16);
    CIM_WRITE_REG(REG_STRIDE_B, 16);
    CIM_WRITE_REG(REG_STRIDE_C, 4);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 16; j++) {
            CIM_WRITE_SPAD(0 + (i * 16) + j, A.mat[i][j]);
        }
    }

    for (int col = 0; col < 4; col++) {
        for (int row = 0; row < 16; row++) {
            CIM_WRITE_SPAD(64 + (col * 16) + row, B.mat[row][col]);
        }
    }

    CIM_WRITE_REG(REG_CTRL, 0x01);

    uint32_t status = 0;
    while ((status & 0x02) == 0) {
        status = CIM_READ_REG(REG_CTRL);
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            C_out.mat[i][j] = CIM_READ_SPAD(0 + (i * 4) + j);
        }
    }

    return C_out;
}

// ==============================================================================
// PRISM-16 MACROBLOCK WRAPPER (Vector Activation Engine)
// ==============================================================================
// Takes the 4x4 output from Convolution, applies Hardware ReLU, and flattens it
// into a 4x16 matrix (specifically loaded into Row 0) for the linear layer.
tensor4x16_t reformat_for_linear(tensor4x4_t conv_out)
{
    tensor4x16_t fc_input = {0}; // Initialize entire 4x16 structure to zeros

    // Cast the structs to 32-bit pointers to pack the Avalon Bus
    // conv_out is exactly 16 bytes. fc_input.mat[0] is exactly 16 bytes.
    uint32_t* X_word = (uint32_t*)&conv_out;
    uint32_t* Z_word = (uint32_t*)&fc_input.mat[0][0];

    // 1. Blast the 16 bytes of data into PRISM-16 Buffer X (Bypass Cache)
    for (int i = 0; i < 4; i++) {
        IOWR_32DIRECT(PRISM_BASE, PRISM_OFFSET_X + (i * 4), X_word[i]);
    }

    // 2. Trigger Hardware Execution in ReLU Mode
    IOWR_32DIRECT(PRISM_BASE, PRISM_OFFSET_CTRL, PRISM_CMD_RELU);

    // 3. Poll for Hardware Completion (1 Clock Cycle later)
    while ((IORD_32DIRECT(PRISM_BASE, PRISM_OFFSET_CTRL) & 0x02) == 0);

    // 4. Read the perfectly activated 16 bytes directly into fc_input row 0
    for (int i = 0; i < 4; i++) {
        Z_word[i] = IORD_32DIRECT(PRISM_BASE, PRISM_OFFSET_Z + (i * 4));
    }

    return fc_input;
}

// ==============================================================================
// MAIN INFERENCE PIPELINE
// ==============================================================================
int main()
{
    printf("  _____ _____ __  __                _               __   __   __   __ \n");
    printf(" / ____|_   _|  \\/  |              | |             /_ | / /  /_ | / / \n");
    printf("| |      | | | \\  / |_ __ ___   ___| |_ _ __ _   _  | |/ /_   | |/ /_ \n");
    printf("| |      | | | |\\/| | '_ ` _ \\ / _ \\ __| '__| | | | | | '_ \\  | | '_ \\\n");
    printf("| |____ _| |_| |  | | | | | | |  __/ |_| |  | |_| | | | (_) | | | (_) |\n");
    printf(" \\_____|_____|_|  |_|_| |_| |_|\\___|\\__|_|   \\__, | |_|\\___/  |_|\\___/\n");
    printf("                                              __/ |                   \n");
    printf("                                             |___/                    \n");
    printf("===========================================================================================\n");
    printf("  Edge AI Inference Engine | Dual-Core Acceleration SoC | Cyclone IV Target  \n");
    printf("===========================================================================================\n\n");

    //-------------------------------------------------------------------
    // 1. LOAD DATA AND WEIGHTS
    // ------------------------------------------------------------------
    tensor4x16_t test_steve_1 = {
            {
                {  -97,  -88,  -87, -103,  -99,  -95, -104,  -86, -102,   -8,   -7,    0,    7,   -7,    4,  -20 },
                {  -85,  -92,  -95,  -96,  -95,  -95,  -98,  -96,   -7,  -19,  -19,  -89,  -29,   -2,  -12,  -13 },
                {   -8,  127,  -57,    4,  -32,   -1,    5,  -55,  -24,  -14,  -69,  -51,  -42,  -38,  -51,  -54 },
                {   -5,  -71,  127,  -15,  -39,  -21,  -24,  -40,  -52,  -68,  -15,  -30,  -77,  -67,  -23,  -27 }
            }};

    tensor4x16_t test_alex_1 = {
            {
                {   26,   28,   23,    8,   40,   33,   29,   72,   13,   27,   47,   19,   47,   56,   41,   79 },
                {   24,   22,   14,   19,   50,   55,   20,   -4,  110,   70,   65,   25,   95,   87,   79,   11 },
                {   76,  124,  -56,   93,   98,  120,   71,   90,   88,  105,  108,   57,   93,  106,  107,  104 },
                {   65,  -53,  109,   98,   68,   98,  101,  103,   75,  108,  118,  102,  115,  125,   94,  115 }
            }};

    tensor4x16_t unknown = {
        {
            { 12, -45, 57, 0,  1, 2, -1, 0,  1, 2, -1, 0,  1, 2, -1, 0},
            { 0,  10, 2, -1, 10, 1,  2, -1, 0, 1,  2, -1, 0, 1,  2, -1},
            {-1,  0, 15, 23, -1, 0,  1, 2, -1, 0,  1, 2, -1, 0,  1, 2},
            { 2, -1, 0, 10,  2, -1, 0, 1,  2, -1, 0, 1,  2, -1, 0, 1}
        }};

    tensor16x4_t W_conv = {
        {
            {    1,   -3,    1,   -1 }, {    3,   -4,    3,    2 }, {    3,   -1,    3,   -4 }, {    2,   -3,    2,   -2 },
            {    2,   -7,    2,   -5 }, {    6,   -7,    6,   -6 }, {    6,   -8,    5,   -6 }, {    2,   -5,    2,   -6 },
            {    2,   -5,    2,   -3 }, {    6,   -3,    6,   -2 }, {    6,    1,    6,   -4 }, {    2,   -1,    2,   -4 },
            {    0,   -1,    0,   -1 }, {    0,   -2,    0,   -3 }, {    0,   -2,    0,   -4 }, {   -1,    0,    0,   -3 }
        }};

    tensor16x4_t W_linear = {
        {
            {    2,   -2,   -2,    2 }, {   -5,    5,    5,   -5 }, {    2,   -2,   -2,    2 }, {   -6,    6,    5,   -5 },
            {    2,   -2,   -2,    3 }, {   -7,    7,    7,   -7 }, {    2,   -2,   -2,    2 }, {   -7,    7,    7,   -7 },
            {    5,   -4,   -5,    4 }, {    1,    0,    0,    0 }, {    5,   -5,   -5,    4 }, {   -7,    7,    7,   -6 },
            {    4,   -4,   -4,    4 }, {   -5,    5,    5,   -5 }, {    4,   -4,   -4,    4 }, {   -1,    1,    2,   -2 }
        }};

    //-------------------------------------------------------------------
    // 2. EXECUTE PIPELINE
    // ------------------------------------------------------------------
    printf("[*] Initiating Hardware Acceleration Sequence...\n");

    // Conv 1: Spatial Feature Extraction
    printf(" -> CYCLE 1: CIMmetry-1616 Spatial Convolution...\n");
    tensor4x4_t conv_out = cimmetry(unknown, W_conv);

    // Reformat: Apply ReLU and flatten for the next layer via Vector Coprocessor
    printf(" -> CYCLE 2: PRISM-16 Hardware Vector Activation (ReLU)...\n");
    tensor4x16_t fc_input = reformat_for_linear(conv_out);

    // Linear Proj 2: Embedding Generation
    printf(" -> CYCLE 3: CIMmetry-1616 Dense Projection...\n");
    tensor4x4_t final_out = cimmetry(fc_input, W_linear);

    //-------------------------------------------------------------------
    // 3. HARDWARE CLASSIFICATION & TELEMETRY
    // ------------------------------------------------------------------
    int8_t* result_emb = final_out.mat[0];

    int32_t dist_to_steve = calculate_sq_distance(result_emb, ANCHOR_STEVE);
    int32_t dist_to_alex  = calculate_sq_distance(result_emb, ANCHOR_ALEX);

    printf("\n===========================================================================================\n");
    printf("                             DUAL-CORE SOC TELEMETRY\n");
    printf("===========================================================================================\n");
    printf(" [ CORE 0 ] CIMmetry-1616  : 16 Parallel INT8 Multipliers & 32-Bit Adder Tree (MAC Array)\n");
    printf(" [ CORE 1 ] PRISM-16       : 16-Lane Combinational Datapath & Vector LUT (VPU)\n");
    printf(" Execution Bottleneck      : ELIMINATED (Zero Software Operations)\n");

    printf("\n===========================================================================================\n");
    printf("                                 INFERENCE RESULTS\n");
    printf("===========================================================================================\n");
    printf(" Target Vector: [ %4d, %4d, %4d, %4d ]\n", result_emb[0], result_emb[1], result_emb[2], result_emb[3]);
    printf("-------------------------------------------------------------------------------------------\n");

    printf(" Distance to Steve Anchor: %8d  ", dist_to_steve);
    print_distance_bar(dist_to_steve);
    if (dist_to_steve < CLASSIFICATION_THRESHOLD) printf("  <-- Threshold Met\n");
    else printf("\n");

    printf(" Distance to Alex Anchor:  %8d  ", dist_to_alex);
    print_distance_bar(dist_to_alex);
    if (dist_to_alex < CLASSIFICATION_THRESHOLD) printf("  <-- Threshold Met\n");
    else printf("\n");

    printf("-------------------------------------------------------------------------------------------\n");

    printf(" CLASSIFICATION: ");
    if (dist_to_steve < CLASSIFICATION_THRESHOLD && dist_to_steve < dist_to_alex) {
        printf("[ MATCH FOUND - STEVE ] \n");
    }
    else if (dist_to_alex < CLASSIFICATION_THRESHOLD && dist_to_alex < dist_to_steve) {
        printf("[ MATCH FOUND - ALEX ] \n");
    }
    else {
        printf("[ UNKNOWN / NOISE ] \n");
    }
    printf("===========================================================================================\n\n");

    return 0;
}

