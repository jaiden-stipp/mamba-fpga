// Explicit Cyclone IV simple dual-port RAM: synchronous one-cycle read.
// The controller never reads and writes the same location in a cycle.
// Simulation uses the installed vendor model, not a zero-latency RAM stub.
module mamba3_cache_ram #(parameter DEPTH=128)(
    input wire clk,write_enable,read_enable,
    input wire [31:0] write_address,read_address,write_data,
    output wire [31:0] read_data
);
    localparam AW=$clog2(DEPTH);
    altsyncram #(
        .operation_mode("DUAL_PORT"),
        .width_a(32),.widthad_a(AW),.numwords_a(DEPTH),
        .width_b(32),.widthad_b(AW),.numwords_b(DEPTH),
        .outdata_reg_b("UNREGISTERED"),.address_reg_b("CLOCK0"),.rdcontrol_reg_b("CLOCK0"),
        .read_during_write_mode_mixed_ports("DONT_CARE"),
        .intended_device_family("Cyclone IV E"),
        .power_up_uninitialized("TRUE"),.lpm_type("altsyncram")
    ) ram(
        .clock0(clk),.clock1(1'b1),.clocken0(1'b1),.clocken1(1'b1),
        .clocken2(1'b1),.clocken3(1'b1),.aclr0(1'b0),.aclr1(1'b0),
        .address_a(write_address[AW-1:0]),.data_a(write_data),.wren_a(write_enable),
        .address_b(read_address[AW-1:0]),.rden_b(read_enable),.q_b(read_data),
        .wren_b(1'b0),.data_b(32'b0),.rden_a(1'b1),
        .byteena_a(1'b1),.byteena_b(1'b1),.addressstall_a(1'b0),.addressstall_b(1'b0)
    );
endmodule
