`timescale 1ns/1ps
module tb_phase_lab;
    reg clk=0;
    always #10 clk=~clk;
    phase_lab dut(.CLOCK_50(clk));
    initial begin
        #24000;
        $display("RESULT completed=%0d guard_errors=%0d low_errors=%0d cycles=%0d phase_guard=%0d phase_low=%0d",
            dut.completed,dut.errors_guard,dut.errors_low,dut.cycles,dut.phase_guard,dut.phase_low);
        if (dut.completed!==1024 || dut.errors_guard!==0 || dut.errors_low!==0 || dut.cycles!==1028) begin
            $display("FAIL");$finish(1);
        end
        $display("PASS");$finish;
    end
endmodule
// Simulation stand-in for the vendor JTAG bridge; datapath and controller are
// the exact synthesis sources. Reset is asserted before releasing the test.
module altsource_probe #(parameter instance_id="",probe_width=1,source_width=1,
    source_initial_value="1",enable_metastability="YES")(
    input wire [probe_width-1:0] probe,output reg [source_width-1:0] source,
    input wire source_clk,source_ena);
    initial begin source=1;#100;source=0;end
endmodule
