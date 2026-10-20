// Modular phase accumulator and pipelined real 2-D rotation.
// Input increments are already in turns / 2^ACC_BITS. Nonlinear preprocessing
// (tanh, dt multiplication) is intentionally outside this measured core.
// Outputs are signed product sums at scale 127: no hidden /127 normalization.
module phase_rotator #(parameter ACC_BITS=16)(
    input wire clk, reset, in_valid,
    input wire [ACC_BITS-1:0] increment,
    input wire signed [15:0] x_in, y_in,
    output wire out_valid,
    output reg signed [24:0] real_out, imag_out,
    output reg [7:0] address_out,
    output reg [ACC_BITS-1:0] phase
);
    reg signed [7:0] cos_rom[0:255];
    reg signed [7:0] sin_rom[0:255];
    initial begin
        $readmemh("cos.hex",cos_rom);
        $readmemh("sin.hex",sin_rom);
    end
    wire [ACC_BITS-1:0] next_phase=phase+increment;
    wire [7:0] address;
    generate if (ACC_BITS==16) begin: round_phase
        // Round to nearest, ties to even, then wrap address modulo 256.
        wire round_up=(next_phase[7:0]>8'h80) ||
                      ((next_phase[7:0]==8'h80) && next_phase[8]);
        assign address=next_phase[15:8]+round_up;
    end else begin: direct_phase
        assign address=next_phase[7:0];
    end endgenerate
    reg [2:0] valid;
    reg [7:0] address_1,address_2;
    reg signed [15:0] x_1,y_1;
    reg signed [7:0] c_1,s_1;
    reg signed [23:0] xc_2,ys_2,xs_2,yc_2;
    assign out_valid=valid[2];
    // Unconditional registered ROM reads permit M9K inference. Invalid/reset
    // pipeline values are ignored by valid; the ROM output needs no reset.
    always @(posedge clk) begin
        c_1<=cos_rom[address];s_1<=sin_rom[address];
    end
    always @(posedge clk) begin
        if (reset) begin
            phase<=0;valid<=0;
            x_1<=0;y_1<=0;
            xc_2<=0;ys_2<=0;xs_2<=0;yc_2<=0;
            real_out<=0;imag_out<=0;
            address_1<=0;address_2<=0;address_out<=0;
        end else begin
            valid<={valid[1:0],in_valid};
            if (in_valid) begin
                phase<=next_phase;
                x_1<=x_in;y_1<=y_in;address_1<=address;
            end
            xc_2<=x_1*c_1;ys_2<=y_1*s_1;
            xs_2<=x_1*s_1;yc_2<=y_1*c_1;
            address_2<=address_1;
            real_out<={xc_2[23],xc_2}-{ys_2[23],ys_2};
            imag_out<={xs_2[23],xs_2}+{yc_2[23],yc_2};
            address_out<=address_2;
        end
    end
endmodule
