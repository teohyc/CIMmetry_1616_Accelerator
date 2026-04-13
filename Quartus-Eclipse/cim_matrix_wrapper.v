module cim_matrix_wrapper (
    input  wire clk,
    input  wire reset,
    input  wire start,

    input  wire [11:0] base_A,
    input  wire [11:0] base_B,
    input  wire [11:0] base_C,

    input  wire [7:0]  stride_A,
    input  wire [7:0]  stride_B,
    input  wire [7:0]  stride_C,

    input  wire        spad_we,
    input  wire [11:0] spad_addr,
    input  wire signed [7:0] spad_wdata,

    //scratchpad read port
    input  wire        spad_re,
    input  wire [11:0] spad_raddr,
    output reg  signed [7:0] spad_rdata,

    output reg         done
);

    // ============================================================
    // SCRATCHPAD
    // ============================================================
    reg signed [7:0] scratchpad [0:128];

    //registered read 
    always @(posedge clk) begin
        if (spad_re)
            spad_rdata <= scratchpad[spad_raddr];
    end

    // ============================================================
    // SYSTOLIC ARRAY
    // ============================================================
    reg compute_en;
    reg signed [127:0] A0_flat, A1_flat, A2_flat, A3_flat;
    reg signed [127:0] B0_flat, B1_flat, B2_flat, B3_flat;

    wire signed [7:0]
        C00, C01, C02, C03,
        C10, C11, C12, C13,
        C20, C21, C22, C23,
        C30, C31, C32, C33;

    wire systolic_done;

    systolic_4x4_top u_systolic (
        .clk(clk),
        .reset(reset),
        .start(compute_en),

        .A0_flat(A0_flat), .A1_flat(A1_flat),
        .A2_flat(A2_flat), .A3_flat(A3_flat),
        .B0_flat(B0_flat), .B1_flat(B1_flat),
        .B2_flat(B2_flat), .B3_flat(B3_flat),

        .C00(C00), .C01(C01), .C02(C02), .C03(C03),
        .C10(C10), .C11(C11), .C12(C12), .C13(C13),
        .C20(C20), .C21(C21), .C22(C22), .C23(C23),
        .C30(C30), .C31(C31), .C32(C32), .C33(C33),

        .done(systolic_done)
    );

    // ============================================================
    // FSM
    // ============================================================
    localparam IDLE    = 3'd0;
    localparam LOAD    = 3'd1;
    localparam COMPUTE = 3'd2;
    localparam WRITE   = 3'd3;
    localparam DONE_ST = 3'd4;

    reg [2:0] state, next_state;

    always @(posedge clk or posedge reset)
        if (reset) state <= IDLE;
        else       state <= next_state;

    always @(*) begin
        next_state = state;
        case (state)
            IDLE:    if (start) next_state = LOAD;
            LOAD:    next_state = COMPUTE;
            COMPUTE: if (systolic_done) next_state = WRITE;
            WRITE:   next_state = DONE_ST;
            DONE_ST: next_state = IDLE;
        endcase
    end

    // ============================================================
    // COMPUTE ENABLE
    // ============================================================
    always @(posedge clk or posedge reset)
        if (reset)
            compute_en <= 1'b0;
        else if (state == LOAD)
            compute_en <= 1'b1;
        else if (systolic_done)
            compute_en <= 1'b0;

    // ============================================================
    // LOAD MATRICES
    // ============================================================
    integer i;

    always @(posedge clk) begin
        if (state == LOAD) begin
            for (i = 0; i < 16; i = i + 1) begin
                A0_flat[i*8 +: 8] <= scratchpad[base_A + 0*stride_A + i];
                A1_flat[i*8 +: 8] <= scratchpad[base_A + 1*stride_A + i];
                A2_flat[i*8 +: 8] <= scratchpad[base_A + 2*stride_A + i];
                A3_flat[i*8 +: 8] <= scratchpad[base_A + 3*stride_A + i];

                B0_flat[i*8 +: 8] <= scratchpad[base_B + 0*stride_B + i];
                B1_flat[i*8 +: 8] <= scratchpad[base_B + 1*stride_B + i];
                B2_flat[i*8 +: 8] <= scratchpad[base_B + 2*stride_B + i];
                B3_flat[i*8 +: 8] <= scratchpad[base_B + 3*stride_B + i];
            end
        end
    end

    // ============================================================
    // WRITEBACK
    // ============================================================
    always @(posedge clk) begin
        if (spad_we)
            scratchpad[spad_addr] <= spad_wdata;
        else if (state == WRITE) begin
            scratchpad[base_C + 0*stride_C + 0] <= C00;
            scratchpad[base_C + 0*stride_C + 1] <= C01;
            scratchpad[base_C + 0*stride_C + 2] <= C02;
            scratchpad[base_C + 0*stride_C + 3] <= C03;

            scratchpad[base_C + 1*stride_C + 0] <= C10;
            scratchpad[base_C + 1*stride_C + 1] <= C11;
            scratchpad[base_C + 1*stride_C + 2] <= C12;
            scratchpad[base_C + 1*stride_C + 3] <= C13;

            scratchpad[base_C + 2*stride_C + 0] <= C20;
            scratchpad[base_C + 2*stride_C + 1] <= C21;
            scratchpad[base_C + 2*stride_C + 2] <= C22;
            scratchpad[base_C + 2*stride_C + 3] <= C23;

            scratchpad[base_C + 3*stride_C + 0] <= C30;
            scratchpad[base_C + 3*stride_C + 1] <= C31;
            scratchpad[base_C + 3*stride_C + 2] <= C32;
            scratchpad[base_C + 3*stride_C + 3] <= C33;
        end
    end

    // ============================================================
    // DONE
    // ============================================================
    always @(posedge clk or posedge reset)
        if (reset) done <= 1'b0;
        else       done <= (state == DONE_ST);

endmodule

