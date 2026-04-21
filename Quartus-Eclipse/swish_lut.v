module swish_lut (
    input  wire       clk,
    input  wire [7:0] addr,
    output reg  [7:0] data
);
    // 256 byte ROM
    reg [7:0] rom [0:255];

    initial begin
        // loads the python-generated answers during synthesis
        $readmemh("swish_values.hex", rom);
    end

    // XOR the highest bit to map signed [-128, 127] to unsigned [0, 255]
    always @(posedge clk) begin
        data <= rom[addr ^ 8'h80];
    end
endmodule