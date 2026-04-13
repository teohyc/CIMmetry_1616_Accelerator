/*
 * main.c
 *
 *  Created on: Jan 22, 2026
 *      Author: TYC
 */
#include <stdio.h>
#include <stdint.h>
#include "io.h"
#include "system.h"

/* Replace this with your actual base address */
#define CIM_BASE  CIM_4X4_VER3_0_BASE

/* Register offsets (word addressing) */
#define REG_START       0
#define REG_DONE        1
#define REG_IN0         2
#define REG_IN1         3
#define REG_IN2         4
#define REG_IN3         5
#define REG_W0          6
#define REG_W1          7
#define REG_W2          8
#define REG_W3          9
#define REG_BIAS        10
#define REG_OUTPUT      11

/* Helper macro */
#define CIM_WRITE(reg, val) IOWR_32DIRECT(CIM_BASE, (reg << 2), (val))
#define CIM_READ(reg)       IORD_32DIRECT(CIM_BASE, (reg << 2))

int main()
{
    int i;
    int32_t done;
    int32_t output;

    /* --------------------------------------
     * Test data (simple but non-trivial)
     * -------------------------------------- */
    int8_t input[16] = {
        1,  2,  -3,  4,
        5,  6,  7,  8,
        -1, -2, -3, -4,
        2,  2,  2,  2
    };

    int8_t weight[16] = {
        1,  1,  1,  1,
        1,  1,  1,  1,
        1,  1,  1,  1,
        1,  1,  1,  1
    };

    int32_t bias = 0;

    printf("=== CIM 4x4 Accelerator Test ===\n");

    /* --------------------------------------
     * Load inputs (4 int8 per write)
     * -------------------------------------- */
    CIM_WRITE(REG_IN0,
        ((uint32_t)(input[3] & 0xFF) << 24) |
        ((uint32_t)(input[2] & 0xFF) << 16) |
        ((uint32_t)(input[1] & 0xFF) << 8)  |
         ((uint32_t)(input[0] & 0xFF)));

    CIM_WRITE(REG_IN1,
        ((uint32_t)(input[7] & 0xFF) << 24) |
        ((uint32_t)(input[6] & 0xFF) << 16) |
        ((uint32_t)(input[5] & 0xFF) << 8)  |
         ((uint32_t)(input[4] & 0xFF)));

    CIM_WRITE(REG_IN2,
        ((uint32_t)(input[11] & 0xFF) << 24) |
        ((uint32_t)(input[10] & 0xFF) << 16) |
        ((uint32_t)(input[9] & 0xFF) << 8)   |
         ((uint32_t)(input[8] & 0xFF)));

    CIM_WRITE(REG_IN3,
        ((uint32_t)(input[15] & 0xFF) << 24) |
        ((uint32_t)(input[14] & 0xFF) << 16) |
        ((uint32_t)(input[13] & 0xFF) << 8)  |
         ((uint32_t)(input[12] & 0xFF)));

    /* --------------------------------------
     * Load weights
     * -------------------------------------- */
    CIM_WRITE(REG_W0, 0x01010101);
    CIM_WRITE(REG_W1, 0x01010101);
    CIM_WRITE(REG_W2, 0x01010101);
    CIM_WRITE(REG_W3, 0x01010101);

    /* --------------------------------------
     * Load bias
     * -------------------------------------- */
    CIM_WRITE(REG_BIAS, bias);

    /* --------------------------------------
     * Clear done flag
     * -------------------------------------- */
    CIM_WRITE(REG_DONE, 1);

    /* --------------------------------------
     * Start computation
     * -------------------------------------- */
    CIM_WRITE(REG_START, 1);

    /* --------------------------------------
     * Poll done
     * -------------------------------------- */
    do {
        done = CIM_READ(REG_DONE) & 0x1;
    } while (done == 0);

    /* --------------------------------------
     * Read output
     * -------------------------------------- */
    output = (int8_t)(CIM_READ(REG_OUTPUT) & 0xFF);

    printf("Output = %d\n", output);

    /* --------------------------------------
     * Software reference
     * -------------------------------------- */
    int32_t sw_sum = 0;
    for (i = 0; i < 16; i++)
        sw_sum += input[i] * weight[i];

    if (sw_sum > 127) sw_sum = 127;
    if (sw_sum < -128) sw_sum = -128;

    printf("Software expected = %d\n", sw_sum);

    while (1);
}


