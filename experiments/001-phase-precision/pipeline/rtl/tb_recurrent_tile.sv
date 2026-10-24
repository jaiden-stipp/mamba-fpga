`timescale 1ns/1ps
module tb_recurrent_tile;
    parameter N=128,P=8,RANK=4,TOKENS=32;
    reg clk=0;always #10 clk=~clk;
    reg reset=1,start_valid=0,v_valid=0,bc_valid=0,y_ready=0;
    wire start_ready,v_ready,bc_ready,y_valid,done,sw;
    wire [31:0] y,si,wide;wire [15:0] yi,packed_state;wire [3:0] stage;
    reg [31:0] alpha,beta,gamma,skip,v,qx,qy,kx,ky;
    reg [15:0] increment;
    mamba3_recurrent_tile #(.N(N),.P(P),.RANK(RANK)) dut(
        .clk(clk),.reset(reset),.start_valid(start_valid),.start_ready(start_ready),
        .alpha_in(alpha),.beta_in(beta),.gamma_in(gamma),.skip_in(skip),
        .v_valid(v_valid),.v_ready(v_ready),.v_in(v),
        .bc_valid(bc_valid),.bc_ready(bc_ready),.qx_in(qx),.qy_in(qy),.kx_in(kx),.ky_in(ky),
        .increment_in(increment),.y_valid(y_valid),.y_ready(y_ready),.y_out(y),.y_index(yi),
        .done(done),.state_write_valid(sw),.state_write_index(si),.state_write_data(packed_state),
        .state_write_wide(wide),.debug_stage(stage));
    reg [127:0] commands[0:TOKENS-1];
    reg [31:0] values[0:TOKENS*P*RANK-1];
    reg [143:0] bc[0:TOKENS*RANK*N/2-1];
    reg [47:0] states[0:TOKENS*P*N-1];
    reg [31:0] outputs[0:TOKENS*P*RANK-1];
    integer t,i,cycles=0,begin_cycle,errors=0,writes=0,reads=0,fd;
    integer bubble_count=0,stall_count=0,last_write_cycle=-1,consecutive=0;
    reg monitor=0,held=0;reg [31:0] held_y;reg [15:0] held_index;
    reg [47:0] expected;
    always @(posedge clk) begin
        cycles=cycles+1;
        if(monitor && !reset) begin
            if(dut.cache_read_enable && dut.previous_write_enable &&
               dut.cache_read_index==dut.previous_write_index) begin
                $display("MISMATCH previous-K RAM read/write collision");errors=errors+1;
            end
            if(sw && stage==6 && si==dut.issue) begin
                $display("MISMATCH state RAM read/write collision");errors=errors+1;
            end
            if(sw) begin
                expected=states[t*P*N+si];
                if($test$plusargs("CORRUPT") && t==0 && si==0) expected=expected^1;
                if(si!==writes% (P*N) || {wide,packed_state}!==expected) begin
                    if(errors<10) $display("MISMATCH state token=%0d index=%0d got=%h%h expected=%h",t,si,wide,packed_state,expected);
                    errors=errors+1;
                end
                $fdisplay(fd,"state,%0d,%0d,%h,%h,%0d",t,si,wide,packed_state,cycles-begin_cycle);
                writes=writes+1;
                if(last_write_cycle==cycles-1) consecutive=consecutive+1;
                last_write_cycle=cycles;
            end
            if(y_valid) begin
                if(held && (y!==held_y || yi!==held_index)) begin
                    $display("MISMATCH output changed while stalled");errors=errors+1;
                end
                held=!y_ready;held_y=y;held_index=yi;
                if(!y_ready) stall_count=stall_count+1;
                if(y_ready) begin
                    if(yi!==reads%(P*RANK) || y!==outputs[t*P*RANK+yi]) begin
                        if(errors<10) $display("MISMATCH y token=%0d index=%0d got=%h expected=%h",t,yi,y,outputs[t*P*RANK+yi]);
                        errors=errors+1;
                    end
                    $fdisplay(fd,"output,%0d,%0d,%h,0000,%0d",t,yi,y,cycles-begin_cycle);
                    reads=reads+1;
                end
            end else if(held) begin $display("MISMATCH dropped stalled output");errors=errors+1;held=0;end
        end
    end
    task clear_core;
        begin
            @(negedge clk);reset=1;start_valid=0;v_valid=0;bc_valid=0;y_ready=0;
            repeat(3) @(negedge clk);
            reset=0;
            while(!start_ready) @(negedge clk);
        end
    endtask
    task load_token(input integer token, input integer bubbles);
        begin
            {alpha,beta,gamma,skip}=commands[token];
            start_valid=1;begin_cycle=cycles;
            @(negedge clk);start_valid=0;
            for(i=0;i<P*RANK;i=i+1) begin
                v_valid=0;
                if(bubbles && i%7==2) begin @(negedge clk);bubble_count=bubble_count+1;end
                while(!v_ready) @(negedge clk);
                v=values[token*P*RANK+i];v_valid=1;
                @(negedge clk);
            end
            v_valid=0;
            for(i=0;i<RANK*N/2;i=i+1) begin
                bc_valid=0;
                if(bubbles && i%11==3) begin @(negedge clk);bubble_count=bubble_count+1;end
                while(!bc_ready) @(negedge clk);
                {qx,qy,kx,ky,increment}=bc[token*RANK*N/2+i];bc_valid=1;
                @(negedge clk);
            end
            bc_valid=0;
        end
    endtask
    initial begin
        $readmemh("commands.hex",commands);$readmemh("values.hex",values);
        $readmemh("bc.hex",bc);$readmemh("states.hex",states);$readmemh("outputs.hex",outputs);
        fd=$fopen("rtl_results.csv","w");$fdisplay(fd,"kind,token,index,wide,packed,cycle");
        clear_core();
        // Abort a real in-flight state update, then prove reset flushes it.
        load_token(0,1);
        while(stage!=6) @(negedge clk);
        repeat(10) @(negedge clk);
        clear_core();monitor=1;
        for(t=0;t<TOKENS;t=t+1) begin
            // Independent fresh sequence in second half, with persistent
            // feedback across every token inside each half.
            if(t==TOKENS/2) begin monitor=0;clear_core();monitor=1;end
            load_token(t,t%2);
            while(!done) begin
                y_ready=(t%2==0) || (cycles%5>=2);
                @(negedge clk);
            end
            $display("TOKEN token=%0d cycles=%0d",t,cycles-begin_cycle);
            y_ready=0;
            @(negedge clk);
        end
        $fclose(fd);
        if(writes!=TOKENS*P*N || reads!=TOKENS*P*RANK) errors=errors+1;
        $display("RESULT rank=%0d tokens=%0d writes=%0d outputs=%0d errors=%0d bubbles=%0d output_stalls=%0d consecutive_write_pairs=%0d",RANK,TOKENS,writes,reads,errors,bubble_count,stall_count,consecutive);
        if(errors==0) $display("PASS");else $display("FAIL");
        $finish;
    end
    initial begin #20000000;$display("FAIL timeout");$finish;end
endmodule
