`timescale 1ns/1ps
module tb_phase_lab #(parameter VECTOR_COUNT=1024);
    reg clk=0;
    always #10 clk=~clk;
    phase_lab #(.VECTOR_COUNT(VECTOR_COUNT)) dut(.CLOCK_50(clk));
    integer result_fd=0;
    integer result_index=0;
    reg [2047:0] result_path;
    initial begin
        if ($value$plusargs("RTL_RESULTS=%s",result_path)) begin
            result_fd=$fopen(result_path,"w");
            if (!result_fd) $fatal(1,"FAIL cannot open RTL results");
            $fdisplay(result_fd,"index,guard_real,guard_imag,guard_address,low_real,low_imag,low_address");
        end
    end
    // Sample between active edges, after nonblocking pipeline updates settle.
    always @(negedge clk) begin
        if (dut.valid_guard===1'b1) begin
            if (dut.valid_low!==1'b1) $fatal(1,"FAIL valid alignment");
            if (result_fd) $fdisplay(result_fd,"%0d,%0d,%0d,%0d,%0d,%0d,%0d",
                result_index,dut.rg,dut.ig,dut.ag,dut.rl,dut.il,dut.al);
            result_index=result_index+1;
        end
        if (dut.valid_low===1'b1 && dut.valid_guard!==1'b1)
            $fatal(1,"FAIL unexpected low-core output");
    end
    initial begin
        #((VECTOR_COUNT+32)*20);
        $display("RESULT completed=%0d guard_errors=%0d low_errors=%0d cycles=%0d phase_guard=%0d phase_low=%0d",
            dut.completed,dut.errors_guard,dut.errors_low,dut.cycles,dut.phase_guard,dut.phase_low);
        if (result_fd) $fclose(result_fd);
        if (dut.completed!==VECTOR_COUNT || dut.errors_guard!==0 || dut.errors_low!==0 ||
            dut.cycles!==(VECTOR_COUNT+4) || result_index!==VECTOR_COUNT) begin
            $fatal(1,"FAIL count, quantized arithmetic, or latency mismatch");
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
