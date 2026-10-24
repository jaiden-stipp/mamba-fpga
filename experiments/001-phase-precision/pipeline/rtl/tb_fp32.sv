`timescale 1ns/1ps
module tb_fp32;
    `include "fp32_functions.svh"
    parameter COUNT=95536;
    reg [191:0] vectors[0:COUNT-1];
    reg [31:0] a,b,add_expected,mul_expected,decoded;
    reg [15:0] half_expected,h;
    integer i,errors;
    initial begin
        $readmemh("fp_vectors.hex",vectors);errors=0;
        for(i=0;i<COUNT;i=i+1) begin
            {a,b,add_expected,mul_expected,half_expected,h,decoded}=vectors[i];
            if(fp_add(a,b)!==add_expected || fp_mul(a,b)!==mul_expected ||
               float_to_half(a)!==half_expected || half_to_float(h)!==decoded) begin
                if(errors<10) $display("MISMATCH i=%0d a=%h b=%h add=%h expected=%h mul=%h expected=%h half=%h expected=%h decode=%h expected=%h",i,a,b,fp_add(a,b),add_expected,fp_mul(a,b),mul_expected,float_to_half(a),half_expected,half_to_float(h),decoded);
                errors=errors+1;
            end
        end
        $display("RESULT arithmetic_vectors=%0d errors=%0d",COUNT,errors);
        if(errors==0) $display("PASS");else $display("FAIL");
        $finish;
    end
endmodule
