module cimmetry1616_1 (
    // Avalon Clock & Reset
    input  wire        csi_clk,
    input  wire        csi_reset_n,  // Avalon uses active-low reset

    // Avalon-MM Slave Interface
    input  wire [12:0] avs_address,  // 13-bit word address
    input  wire        avs_read,
    output wire [31:0] avs_readdata,
    input  wire        avs_write,
    input  wire [31:0] avs_writedata
);

    // ------------------------------------------------------------
    // Internal Registers for Configuration
    // ------------------------------------------------------------
    reg [11:0] reg_base_A, reg_base_B, reg_base_C;
    reg [7:0]  reg_stride_A, reg_stride_B, reg_stride_C;
    
    reg        start_pulse;
    reg        done_flag; // Latches the IP's 1-cycle done signal
    wire       ip_done;

    // Active-high reset for the inner IP
    wire ip_reset = ~csi_reset_n;

    // ------------------------------------------------------------
    // IP Instantiation
    // ------------------------------------------------------------
    wire [11:0] ip_spad_addr;
    wire [7:0]  ip_spad_wdata;
    wire        ip_spad_we;
    wire        ip_spad_re;
    wire signed [7:0] ip_spad_rdata;

    cim_matrix_wrapper u_cim_core (
        .clk        (csi_clk),
        .reset      (ip_reset),
        .start      (start_pulse),

        .base_A     (reg_base_A),
        .base_B     (reg_base_B),
        .base_C     (reg_base_C),

        .stride_A   (reg_stride_A),
        .stride_B   (reg_stride_B),
        .stride_C   (reg_stride_C),

        .spad_we    (ip_spad_we),
        .spad_addr  (ip_spad_addr),
        .spad_wdata (ip_spad_wdata),

        .spad_re    (ip_spad_re),
        .spad_raddr (ip_spad_addr), // Shared address bus from Avalon
        .spad_rdata (ip_spad_rdata),

        .done       (ip_done)
    );

    // ------------------------------------------------------------
    // Write Logic & Control
    // ------------------------------------------------------------
    always @(posedge csi_clk or negedge csi_reset_n) begin
        if (~csi_reset_n) begin
            reg_base_A   <= 12'd0;
            reg_base_B   <= 12'd64;
            reg_base_C   <= 12'd96;
            reg_stride_A <= 8'd16;
            reg_stride_B <= 8'd16;
            reg_stride_C <= 8'd4;
            start_pulse  <= 1'b0;
            done_flag    <= 1'b0;
        end else begin
            // Default start to 0 to create a 1-cycle pulse
            start_pulse <= 1'b0;

            // Latch Done signal from IP. Clear it when starting a new operation.
            if (ip_done)
                done_flag <= 1'b1;
            else if (start_pulse)
                done_flag <= 1'b0;

            // Register Writes (Address bit 12 == 0)
            if (avs_write && !avs_address[12]) begin
                case (avs_address[3:0])
                    4'd0: start_pulse  <= avs_writedata[0]; // Pulse start
                    4'd1: reg_base_A   <= avs_writedata[11:0];
                    4'd2: reg_base_B   <= avs_writedata[11:0];
                    4'd3: reg_base_C   <= avs_writedata[11:0];
                    4'd4: reg_stride_A <= avs_writedata[7:0];
                    4'd5: reg_stride_B <= avs_writedata[7:0];
                    4'd6: reg_stride_C <= avs_writedata[7:0];
                endcase
            end
        end
    end

    // ------------------------------------------------------------
    // Scratchpad Memory Mapping
    // ------------------------------------------------------------
    // If bit 12 is high, we are targeting the scratchpad
    assign ip_spad_we    = avs_write & avs_address[12];
    assign ip_spad_re    = avs_read  & avs_address[12];
    assign ip_spad_addr  = avs_address[11:0];
    assign ip_spad_wdata = avs_writedata[7:0];

    // ------------------------------------------------------------
    // Read Logic (Registers & Memory)
    // ------------------------------------------------------------
    // We register the read data for standard registers to match the 
    // 1-cycle latency of the scratchpad SRAM.
    reg [31:0] reg_readdata;
    reg        read_is_mem;

    always @(posedge csi_clk) begin
        if (avs_read) begin
            read_is_mem <= avs_address[12];
            case (avs_address[3:0])
                4'd0: reg_readdata <= {30'd0, done_flag, 1'b0};
                4'd1: reg_readdata <= {20'd0, reg_base_A};
                4'd2: reg_readdata <= {20'd0, reg_base_B};
                4'd3: reg_readdata <= {20'd0, reg_base_C};
                4'd4: reg_readdata <= {24'd0, reg_stride_A};
                4'd5: reg_readdata <= {24'd0, reg_stride_B};
                4'd6: reg_readdata <= {24'd0, reg_stride_C};
                default: reg_readdata <= 32'd0;
            endcase
        end
    end

    // Sign-extend the 8-bit scratchpad data to 32-bit for the Nios bus
    wire [31:0] spad_readdata_ext = {{24{ip_spad_rdata[7]}}, ip_spad_rdata};

    // Mux between Memory and Register reads
    assign avs_readdata = read_is_mem ? spad_readdata_ext : reg_readdata;

endmodule
		