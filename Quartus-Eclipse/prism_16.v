// PRISM-16: Parallel ReLU & Integer SwiGLU Macroblock


module prism_16 (
    //Avalon Clock & Active-Low Reset
    input  wire        csi_clk,
    input  wire        csi_reset_n,

    //Avalon-MM 
    input  wire [3:0]  avs_address,
    input  wire        avs_read,
    output reg  [31:0] avs_readdata,
    input  wire        avs_write,
    input  wire [31:0] avs_writedata
);

    // ------------------------------------------------------------
    // INTERNAL BUFFER REGISTERS
    // ------------------------------------------------------------
    // 16 Elements = 16 Bytes Each
    reg signed [7:0] reg_X [0:15];
    reg signed [7:0] reg_Y [0:15];
    reg signed [7:0] reg_Z [0:15];

    reg       done_flag;
    reg [1:0] op_mode;      // Stores the requested operation mode
    reg       compute_pulse;
    reg [1:0] pipe;         // Delay pipeline for LUT memory read

    // ------------------------------------------------------------
    // PRISM-16 DATAPATH (16 Parallel Vector Lanes)
    // ------------------------------------------------------------
    wire signed [7:0]  swish_out [0:15];
    wire signed [7:0]  relu_out  [0:15];
    wire signed [15:0] mult_out  [0:15];
    wire signed [7:0]  sat_Z     [0:15];
    wire signed [7:0]  final_Z   [0:15];

    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1) begin : PRISM_LANE
            // 1. Swish LUT (Instantiates M9K block via sub-module)
            swish_lut lut (
                .clk (csi_clk),
                .addr(reg_X[i]),
                .data(swish_out[i])
            );

            // 2. Hardware ReLU (Combinational)
            assign relu_out[i] = (reg_X[i] > 0) ? reg_X[i] : 8'sd0;

            // 3. DSP Multiplier (Muxes operand A based on SwiGLU vs Normal Multiply)
            assign mult_out[i] = ((op_mode == 2'b11) ? swish_out[i] : reg_X[i]) * reg_Y[i];
            
            // 4. Bit-Shift Scaling (Divide by 64 to match QAT)
            wire signed [15:0] shifted = mult_out[i] >>> 6;

            // 5. Multiplier Saturation Clamp
            assign sat_Z[i] = 
                (shifted > 127)  ?  8'sd127 :
                (shifted < -128) ? -8'sd128 :
                                    shifted[7:0];

            // 6. Final Output Multiplexer (Routes data based on Opcode)
            assign final_Z[i] = 
                (op_mode == 2'b00) ? relu_out[i] :  // Opcode 00: ReLU
                (op_mode == 2'b01) ? swish_out[i] : // Opcode 01: Swish
                                     sat_Z[i];      // Opcode 10/11: Multiply/SwiGLU
        end
    endgenerate

    // ------------------------------------------------------------
    // WRITE LOGIC & 1-CYCLE COMPUTE PIPELINE
    // ------------------------------------------------------------
    integer k;
    always @(posedge csi_clk or negedge csi_reset_n) begin
        if (~csi_reset_n) begin
            done_flag <= 1'b0;
            compute_pulse <= 1'b0;
            pipe <= 2'b00;
            op_mode <= 2'b00;
            for (k = 0; k < 16; k = k + 1) begin
                reg_X[k] <= 8'sd0;
                reg_Y[k] <= 8'sd0;
                reg_Z[k] <= 8'sd0;
            end
        end else begin
            // Shift register to track the 2-cycle compute delay (1 for LUT, 1 to latch)
            compute_pulse <= 1'b0;
            pipe <= {pipe[0], compute_pulse};

            // Latch the settled combinational math instantly when pipe[1] fires
            if (pipe[1]) begin
                for (k = 0; k < 16; k = k + 1) begin
                    reg_Z[k] <= final_Z[k];
                end
                done_flag <= 1'b1;
            end

            // Avalon 32-bit Writes (Little Endian Byte Packing)
            if (avs_write) begin
                case (avs_address)
                    // Write to Buffer X
                    4'd0: {reg_X[3], reg_X[2], reg_X[1], reg_X[0]}   <= avs_writedata;
                    4'd1: {reg_X[7], reg_X[6], reg_X[5], reg_X[4]}   <= avs_writedata;
                    4'd2: {reg_X[11],reg_X[10],reg_X[9], reg_X[8]}   <= avs_writedata;
                    4'd3: {reg_X[15],reg_X[14],reg_X[13],reg_X[12]}  <= avs_writedata;
                    
                    // Write to Buffer Y
                    4'd4: {reg_Y[3], reg_Y[2], reg_Y[1], reg_Y[0]}   <= avs_writedata;
                    4'd5: {reg_Y[7], reg_Y[6], reg_Y[5], reg_Y[4]}   <= avs_writedata;
                    4'd6: {reg_Y[11],reg_Y[10],reg_Y[9], reg_Y[8]}   <= avs_writedata;
                    4'd7: {reg_Y[15],reg_Y[14],reg_Y[13],reg_Y[12]}  <= avs_writedata;
                    
                    // Write to Control Register
                    4'd8: begin
                        if (avs_writedata[0]) begin
                            op_mode <= avs_writedata[2:1]; // Capture Opcode
                            compute_pulse <= 1'b1;         // Trigger pipeline
                            done_flag <= 1'b0;             // Clear done flag
                        end
                    end
                endcase
            end
        end
    end

    // ------------------------------------------------------------
    // AVALON READ MULTIPLEXER
    // ------------------------------------------------------------
    always @(posedge csi_clk) begin
        if (avs_read) begin
            case (avs_address)
                // Read Control Status
                4'd8:  avs_readdata <= {30'd0, done_flag, 1'b0};
                
                // Read Buffer Z Output
                4'd12: avs_readdata <= {reg_Z[3], reg_Z[2], reg_Z[1], reg_Z[0]};
                4'd13: avs_readdata <= {reg_Z[7], reg_Z[6], reg_Z[5], reg_Z[4]};
                4'd14: avs_readdata <= {reg_Z[11],reg_Z[10],reg_Z[9], reg_Z[8]};
                4'd15: avs_readdata <= {reg_Z[15],reg_Z[14],reg_Z[13],reg_Z[12]};
                
                default: avs_readdata <= 32'd0;
            endcase
        end
    end
endmodule