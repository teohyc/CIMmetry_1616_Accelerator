// ============================================================
// CIM 4x4 CNN MAC Accelerator (Compute-only, Verilog compatible)
// - int8 inputs & weights (16 lanes)
// - int32 accumulation
// - int8 saturated output
// ============================================================

module cim_4x4_ver3_compute (
    input  wire        clk,
    input  wire        reset,

    input  wire        compute,
    input  wire signed [127:0] input_vec_flat,   // 16 x int8
    input  wire signed [127:0] weight_vec_flat,  // 16 x int8
    input  wire signed [31:0]  bias,

    output wire        done,
    output wire signed [7:0]  output_val
);

    // -------------------------------------------------
    // Unpack vectors (manual, Verilog-safe)
    // -------------------------------------------------
    wire signed [7:0] input_vec  [0:15];
    wire signed [7:0] weight_vec [0:15];

    assign input_vec[0]  = input_vec_flat[7:0];
    assign input_vec[1]  = input_vec_flat[15:8];
    assign input_vec[2]  = input_vec_flat[23:16];
    assign input_vec[3]  = input_vec_flat[31:24];
    assign input_vec[4]  = input_vec_flat[39:32];
    assign input_vec[5]  = input_vec_flat[47:40];
    assign input_vec[6]  = input_vec_flat[55:48];
    assign input_vec[7]  = input_vec_flat[63:56];
    assign input_vec[8]  = input_vec_flat[71:64];
    assign input_vec[9]  = input_vec_flat[79:72];
    assign input_vec[10] = input_vec_flat[87:80];
    assign input_vec[11] = input_vec_flat[95:88];
    assign input_vec[12] = input_vec_flat[103:96];
    assign input_vec[13] = input_vec_flat[111:104];
    assign input_vec[14] = input_vec_flat[119:112];
    assign input_vec[15] = input_vec_flat[127:120];

    assign weight_vec[0]  = weight_vec_flat[7:0];
    assign weight_vec[1]  = weight_vec_flat[15:8];
    assign weight_vec[2]  = weight_vec_flat[23:16];
    assign weight_vec[3]  = weight_vec_flat[31:24];
    assign weight_vec[4]  = weight_vec_flat[39:32];
    assign weight_vec[5]  = weight_vec_flat[47:40];
    assign weight_vec[6]  = weight_vec_flat[55:48];
    assign weight_vec[7]  = weight_vec_flat[63:56];
    assign weight_vec[8]  = weight_vec_flat[71:64];
    assign weight_vec[9]  = weight_vec_flat[79:72];
    assign weight_vec[10] = weight_vec_flat[87:80];
    assign weight_vec[11] = weight_vec_flat[95:88];
    assign weight_vec[12] = weight_vec_flat[103:96];
    assign weight_vec[13] = weight_vec_flat[111:104];
    assign weight_vec[14] = weight_vec_flat[119:112];
    assign weight_vec[15] = weight_vec_flat[127:120];

    // -------------------------------------------------
    // Parallel multipliers
    // -------------------------------------------------
    wire signed [15:0] mult [0:15];

    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1) begin : MUL
            assign mult[i] = input_vec[i] * weight_vec[i];
        end
    endgenerate

    // -------------------------------------------------
    // Adder tree
    // -------------------------------------------------
    wire signed [31:0] sum0  = mult[0]  + mult[1];
    wire signed [31:0] sum1  = mult[2]  + mult[3];
    wire signed [31:0] sum2  = mult[4]  + mult[5];
    wire signed [31:0] sum3  = mult[6]  + mult[7];
    wire signed [31:0] sum4  = mult[8]  + mult[9];
    wire signed [31:0] sum5  = mult[10] + mult[11];
    wire signed [31:0] sum6  = mult[12] + mult[13];
    wire signed [31:0] sum7  = mult[14] + mult[15];

    wire signed [31:0] sum8  = sum0 + sum1;
    wire signed [31:0] sum9  = sum2 + sum3;
    wire signed [31:0] sum10 = sum4 + sum5;
    wire signed [31:0] sum11 = sum6 + sum7;

    wire signed [31:0] sum12 = sum8  + sum9;
    wire signed [31:0] sum13 = sum10 + sum11;

    wire signed [31:0] mac_acc = sum12 + sum13 + bias;

    // -------------------------------------------------
    // Saturation
    // -------------------------------------------------
    wire signed [7:0] mac_sat =
        (mac_acc > 127)  ?  8'sd127 :
        (mac_acc < -128) ? -8'sd128 :
                           mac_acc[7:0];

    // -------------------------------------------------
    // Output register
    // -------------------------------------------------
    reg signed [7:0] output_reg;

    always @(posedge clk or posedge reset) begin
        if (reset)
            output_reg <= 8'sd0;
        else if (compute)
            output_reg <= mac_sat;
    end

    assign output_val = output_reg;
    assign done       = compute;

endmodule
