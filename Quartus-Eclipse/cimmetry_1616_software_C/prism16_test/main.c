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
// PRISM-16 HARDWARE BASE ADDRESS & MEMORY MAP
// ==============================================================================
#define PRISM_BASE PRISM_16_0_BASE


// PRISM-16 Memory Map Offsets (in Bytes)
#define PRISM_OFFSET_X    0   // Buffer X starts at offset 0x00
#define PRISM_OFFSET_Y    16  // Buffer Y starts at offset 0x10
#define PRISM_OFFSET_CTRL 32  // Control Reg is at offset 0x20
#define PRISM_OFFSET_Z    48  // Buffer Z starts at offset 0x30

// ==============================================================================
// PRISM-16 OPCODES
// ==============================================================================
// Mode bits [2:1] shifted left, OR'd with the Trigger Bit 0
#define PRISM_CMD_RELU    ((0 << 1) | 1) // 0x01
#define PRISM_CMD_SWISH   ((1 << 1) | 1) // 0x03
#define PRISM_CMD_MUL     ((2 << 1) | 1) // 0x05
#define PRISM_CMD_SWIGLU  ((3 << 1) | 1) // 0x07

// ==============================================================================
// HARDWARE DRIVER (CACHE-BYPASS)
// ==============================================================================
// Loads 16-byte vectors into PRISM-16, triggers the specified opcode, and reads back
void execute_prism16(int8_t* X_in, int8_t* Y_in, int8_t* Z_out, uint32_t opcode) {

    // Cast 8-bit arrays to 32-bit pointers for maximum Avalon bus bandwidth
    uint32_t* X_word = (uint32_t*)X_in;
    uint32_t* Y_word = (uint32_t*)Y_in;
    uint32_t* Z_word = (uint32_t*)Z_out;

    // 1. Blast X and Y to the hardware, bypassing the Nios II data cache
    for (int i = 0; i < 4; i++) {
        // Offset increases by 4 bytes for every 32-bit word written
        IOWR_32DIRECT(PRISM_BASE, PRISM_OFFSET_X + (i * 4), X_word[i]);
        IOWR_32DIRECT(PRISM_BASE, PRISM_OFFSET_Y + (i * 4), Y_word[i]);
    }

    // 2. Trigger Compute with the selected Opcode
    IOWR_32DIRECT(PRISM_BASE, PRISM_OFFSET_CTRL, opcode);

    // 3. Poll for Done Flag (Wait for bit 1 to go high)
    while ((IORD_32DIRECT(PRISM_BASE, PRISM_OFFSET_CTRL) & 0x02) == 0);

    // 4. Read the 16 results back directly from the hardware
    for (int i = 0; i < 4; i++) {
        Z_word[i] = IORD_32DIRECT(PRISM_BASE, PRISM_OFFSET_Z + (i * 4));
    }
}

// ==============================================================================
// HELPER PRINT FUNCTION
// ==============================================================================
void print_vector(const char* label, int8_t* vec) {
    printf("%-15s: [ ", label);
    for (int i = 0; i < 16; i++) {
        printf("%4d ", vec[i]);
    }
    printf("]\n");
}

// ==============================================================================
// MAIN TEST SUITE
// ==============================================================================
int main() {
    printf("\n================================================================================\n");
    printf("  PRISM-16: Parallel ReLU & Integer SwiGLU Macroblock | Hardware Diagnostics\n");
    printf("================================================================================\n\n");

    // Test Data Setup
    // X spans from highly negative to highly positive to thoroughly test the non-linear curves
    int8_t vector_X[16] = {-128, -80, -40, -10, -5, -1, 0, 1, 5, 10, 20, 40, 64, 80, 100, 127};

    // Y is set exactly to 64.
    // Because the hardware divides by 64 (>>> 6) after multiplying, Y=64 acts as the Identity Multiplier.
    int8_t vector_Y[16] = {64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64};

    int8_t vector_Z[16] = {0};

    printf("--- INPUT VECTORS ---\n");
    print_vector("Vector X (Data)", vector_X);
    print_vector("Vector Y (Scale)", vector_Y);
    printf("--------------------------------------------------------------------------------\n\n");

    // ---------------------------------------------------------
    // TEST 1: RELU MODE (Opcode: 0x01)
    // ---------------------------------------------------------
    printf("[*] Executing Hardware Mode 00: RELU...\n");
    execute_prism16(vector_X, vector_Y, vector_Z, PRISM_CMD_RELU);
    print_vector("Output Z", vector_Z);
    printf("    -> Expectation: All negative numbers clamp to 0. Positives remain unchanged.\n\n");

    // ---------------------------------------------------------
    // TEST 2: SWISH MODE (Opcode: 0x03)
    // ---------------------------------------------------------
    printf("[*] Executing Hardware Mode 01: SWISH LUT...\n");
    execute_prism16(vector_X, vector_Y, vector_Z, PRISM_CMD_SWISH);
    print_vector("Output Z", vector_Z);
    printf("    -> Expectation: Negatives crush to 0. Small negatives dip slightly. Positives scale.\n\n");

    // ---------------------------------------------------------
    // TEST 3: MULTIPLY MODE (Opcode: 0x05)
    // ---------------------------------------------------------
    printf("[*] Executing Hardware Mode 10: ELEMENT-WISE MULTIPLY...\n");
    execute_prism16(vector_X, vector_Y, vector_Z, PRISM_CMD_MUL);
    print_vector("Output Z", vector_Z);
    printf("    -> Expectation: Because Y=64, (X * 64) >> 6 = X. Should perfectly match Vector X.\n\n");

    // ---------------------------------------------------------
    // TEST 4: SWIGLU MODE (Opcode: 0x07)
    // ---------------------------------------------------------
    printf("[*] Executing Hardware Mode 11: SWIGLU...\n");
    execute_prism16(vector_X, vector_Y, vector_Z, PRISM_CMD_SWIGLU);
    print_vector("Output Z", vector_Z);
    printf("    -> Expectation: Because Y=64, (Swish(X) * 64) >> 6 = Swish(X). Should match Swish output.\n\n");

    printf("================================================================================\n");
    printf("  ALL DIAGNOSTICS COMPLETE.\n");
    printf("================================================================================\n\n");

    return 0;
}

