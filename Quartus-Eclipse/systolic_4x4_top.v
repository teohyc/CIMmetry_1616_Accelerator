module systolic_4x4_top (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,

    // A matrix rows (4 × 16)
    input  wire signed [127:0] A0_flat,
    input  wire signed [127:0] A1_flat,
    input  wire signed [127:0] A2_flat,
    input  wire signed [127:0] A3_flat,

    // B matrix columns (4 × 16)
    input  wire signed [127:0] B0_flat,
    input  wire signed [127:0] B1_flat,
    input  wire signed [127:0] B2_flat,
    input  wire signed [127:0] B3_flat,

    // Output C matrix (4 × 4)
    output wire signed [7:0] C00, C01, C02, C03,
    output wire signed [7:0] C10, C11, C12, C13,
    output wire signed [7:0] C20, C21, C22, C23,
    output wire signed [7:0] C30, C31, C32, C33,

    output wire        done
);

    // -----------------------------------------
    // Bias (zero for now)
    // -----------------------------------------
    wire signed [31:0] bias = 32'sd0;

    // Done wires
    wire d[0:15];

    // -----------------------------------------
    // Row 0
    // -----------------------------------------
    cim_4x4_ver3_compute u00 (clk, reset, start, A0_flat, B0_flat, bias, d[0],  C00);
    cim_4x4_ver3_compute u01 (clk, reset, start, A0_flat, B1_flat, bias, d[1],  C01);
    cim_4x4_ver3_compute u02 (clk, reset, start, A0_flat, B2_flat, bias, d[2],  C02);
    cim_4x4_ver3_compute u03 (clk, reset, start, A0_flat, B3_flat, bias, d[3],  C03);

    // -----------------------------------------
    // Row 1
    // -----------------------------------------
    cim_4x4_ver3_compute u10 (clk, reset, start, A1_flat, B0_flat, bias, d[4],  C10);
    cim_4x4_ver3_compute u11 (clk, reset, start, A1_flat, B1_flat, bias, d[5],  C11);
    cim_4x4_ver3_compute u12 (clk, reset, start, A1_flat, B2_flat, bias, d[6],  C12);
    cim_4x4_ver3_compute u13 (clk, reset, start, A1_flat, B3_flat, bias, d[7],  C13);

    // -----------------------------------------
    // Row 2
    // -----------------------------------------
    cim_4x4_ver3_compute u20 (clk, reset, start, A2_flat, B0_flat, bias, d[8],  C20);
    cim_4x4_ver3_compute u21 (clk, reset, start, A2_flat, B1_flat, bias, d[9],  C21);
    cim_4x4_ver3_compute u22 (clk, reset, start, A2_flat, B2_flat, bias, d[10], C22);
    cim_4x4_ver3_compute u23 (clk, reset, start, A2_flat, B3_flat, bias, d[11], C23);

    // -----------------------------------------
    // Row 3
    // -----------------------------------------
    cim_4x4_ver3_compute u30 (clk, reset, start, A3_flat, B0_flat, bias, d[12], C30);
    cim_4x4_ver3_compute u31 (clk, reset, start, A3_flat, B1_flat, bias, d[13], C31);
    cim_4x4_ver3_compute u32 (clk, reset, start, A3_flat, B2_flat, bias, d[14], C32);
    cim_4x4_ver3_compute u33 (clk, reset, start, A3_flat, B3_flat, bias, d[15], C33);

    // -----------------------------------------
    // Done when all CIMs done
    // -----------------------------------------
    assign done =
        d[0]  & d[1]  & d[2]  & d[3]  &
        d[4]  & d[5]  & d[6]  & d[7]  &
        d[8]  & d[9]  & d[10] & d[11] &
        d[12] & d[13] & d[14] & d[15];

endmodule
