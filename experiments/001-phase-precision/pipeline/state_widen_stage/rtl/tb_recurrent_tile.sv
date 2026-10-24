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
    integer rotation_accepts=0,rotation_adds=0,rotation_writes=0;
    integer tag_pair,tag_rank,tag_address,tag_remaining=0;
    integer post_reset_checks=0;
    reg tag_pending=0;
    integer kv_admits=0,kv_consumes=0,kv_tag_q_checks=0;
    reg expected_kv_valid=0;
    integer expected_kv_index;
    reg [31:0] expected_kv_q;
    // Independent one-cycle address/Q monitor for the new product boundary.
    // Numerical results remain checked by the existing external golden files.
    always @(posedge clk) begin
        if(reset || !monitor || RANK!=1) expected_kv_valid<=0;
        else begin
            if(dut.valid_pipe[2]!==expected_kv_valid) begin
                $display("MISMATCH KV product valid");errors=errors+1;
            end
            if(dut.valid_pipe[2]) begin
                if(dut.indices[2]!==expected_kv_index || dut.kv_q!==expected_kv_q) begin
                    $display("MISMATCH KV product tag/Q index=%0d expected=%0d",dut.indices[2],expected_kv_index);
                    errors=errors+1;
                end
                kv_consumes=kv_consumes+1;kv_tag_q_checks=kv_tag_q_checks+1;
            end
            expected_kv_valid<=dut.valid_pipe[1];
            if(dut.valid_pipe[1]) begin
                expected_kv_index<=dut.indices[1];expected_kv_q<=dut.read_q;
                kv_admits=kv_admits+1;
            end
        end
        if(!reset && dut.mode==0 && dut.valid_pipe!==0) begin
            $display("MISMATCH stale state pipeline valid after reset");errors=errors+1;
        end
    end
    integer sum_admits=0,sum_consumes=0;
    reg expected_sum_valid=0;
    integer expected_sum_index;
    reg [31:0] expected_sum_now,expected_sum_q;
    always @(posedge clk) begin
        if(reset || !monitor || RANK!=1) expected_sum_valid<=0;
        else begin
            if(dut.valid_pipe[5]!==expected_sum_valid) begin
                $display("MISMATCH SUM valid");errors=errors+1;
            end
            if(dut.valid_pipe[5]) begin
                if(dut.indices[5]!==expected_sum_index || dut.update_now!==expected_sum_now || dut.update_q!==expected_sum_q) begin
                    $display("MISMATCH SUM tag/now/Q");errors=errors+1;
                end
                sum_consumes=sum_consumes+1;
            end
            expected_sum_valid<=dut.valid_pipe[4];
            if(dut.valid_pipe[4]) begin
                expected_sum_index<=dut.indices[4];expected_sum_now<=dut.now2;expected_sum_q<=dut.q2[0];
                sum_admits=sum_admits+1;
            end
        end
    end
    integer phase_admits=0,phase_consumes=0;
    reg expected_phase_valid=0;
    reg [191:0] expected_phase_words;
    always @(posedge clk) begin
        if(reset || !monitor || RANK!=1) expected_phase_valid<=0;
        else begin
            if((stage==12)!==expected_phase_valid) begin
                $display("MISMATCH PHASE valid");errors=errors+1;
            end
            if(stage==12) begin
                if({dut.phase_qx,dut.phase_qy,dut.phase_kx,dut.phase_ky,dut.phase_c,dut.phase_s}!==expected_phase_words) begin
                    $display("MISMATCH PHASE captured coordinates/trig");errors=errors+1;
                end
                phase_consumes=phase_consumes+1;
            end
            expected_phase_valid<=stage==3 && bc_valid && bc_ready && dut.bpair<N/4;
            if(stage==3 && bc_valid && bc_ready && dut.bpair<N/4) begin
                expected_phase_words<={qx,qy,kx,ky,dut.c,dut.s};phase_admits=phase_admits+1;
            end
        end
    end
    integer rom_admits=0,rom_consumes=0;
    reg expected_rom_valid=0;
    reg [191:0] expected_rom_words;
    always @(posedge clk) begin
        if(reset || !monitor || RANK!=1) expected_rom_valid<=0;
        else begin
            if((stage==11)!==expected_rom_valid) begin
                $display("MISMATCH ROMOUT valid");errors=errors+1;
            end
            if(stage==11) begin
                if({dut.trig_qx,dut.trig_qy,dut.trig_kx,dut.trig_ky,dut.trig_c,dut.trig_s}!==expected_rom_words) begin
                    $display("MISMATCH ROMOUT captured coordinates/trig");errors=errors+1;
                end
                rom_consumes=rom_consumes+1;
            end
            expected_rom_valid<=stage==12;
            if(stage==12) begin
                expected_rom_words<={dut.phase_qx,dut.phase_qy,dut.phase_kx,dut.phase_ky,dut.phase_c,dut.phase_s};
                rom_admits=rom_admits+1;
            end
        end
    end
    integer read_admits=0,read_consumes=0;
    reg expected_read_valid=0;
    integer expected_read_index;
    reg [191:0] expected_read_words;
    always @(posedge clk) begin
        if(reset || !monitor || RANK!=1) expected_read_valid<=0;
        else begin
            if(dut.valid_pipe[1]!==expected_read_valid) begin
                $display("MISMATCH READ valid");errors=errors+1;
            end
            if(dut.valid_pipe[1]) begin
                if(dut.indices[1]!==expected_read_index ||
                   {dut.read_k,dut.read_pk,dut.read_v,dut.read_pv,dut.read_q,dut.read_state}!==expected_read_words) begin
                    $display("MISMATCH READ aligned data/address");errors=errors+1;
                end
                read_consumes=read_consumes+1;
            end
            expected_read_valid<=dut.valid_pipe[0];
            if(dut.valid_pipe[0]) begin
                expected_read_index<=dut.indices[0];
                expected_read_words<={dut.k0[0],dut.pk0[0],dut.v0[0],dut.pv0[0],dut.q0[0],dut.half_to_float(dut.stored0)};
                read_admits=read_admits+1;
            end
        end
    end
    // Check acceptance-to-write tag lifetime independently of the data oracle.
    always @(posedge clk) begin
        if(reset) begin tag_pending=0;tag_remaining=0;end
        else begin
            if(dut.mode==0) begin
                if(dut.rotation_valid!==0) begin
                    $display("MISMATCH stale rotation valid after reset");errors=errors+1;
                end
                if(monitor) post_reset_checks=post_reset_checks+1;
            end
            if(monitor && RANK==1) begin
                if(stage==3 && bc_valid && bc_ready && dut.bpair<N/4) begin
                    if(tag_pending) begin $display("MISMATCH rotation overwritten");errors=errors+1;end
                    tag_pair=dut.bpair;tag_rank=dut.br;tag_address=dut.address;
                    tag_pending=1;tag_remaining=2;rotation_accepts=rotation_accepts+1;
                end
                if(stage==12 || stage==11 || stage==10 || (dut.rotation_valid && (stage==4 || stage==5))) begin
                    if(!tag_pending || dut.rotation_valid!==1 || dut.rotation_pair!==tag_pair ||
                       dut.rotation_rank!==tag_rank || dut.rotation_address!==tag_address || bc_ready) begin
                        $display("MISMATCH rotation tag/valid pair=%0d rank=%0d address=%0d stage=%0d",
                                 dut.rotation_pair,dut.rotation_rank,dut.rotation_address,stage);errors=errors+1;
                    end
                    if(stage==10) rotation_adds=rotation_adds+1;
                    else if(stage==4 || stage==5) begin
                        if(dut.bc_write_index!==(2*tag_pair+(stage==5))) begin
                            $display("MISMATCH rotation write address");errors=errors+1;
                        end
                        rotation_writes=rotation_writes+1;tag_remaining=tag_remaining-1;
                        if(tag_remaining==0) tag_pending=0;
                    end
                end
            end
        end
    end
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
    task first_pair;
        begin
            {alpha,beta,gamma,skip}=commands[0];start_valid=1;
            @(negedge clk);start_valid=0;
            for(i=0;i<P*RANK;i=i+1) begin
                while(!v_ready) @(negedge clk);
                v=values[i];v_valid=1;@(negedge clk);
            end
            v_valid=0;while(!bc_ready) @(negedge clk);
            {qx,qy,kx,ky,increment}=bc[0];bc_valid=1;
            @(negedge clk);bc_valid=0;
        end
    endtask
    initial begin
        $readmemh("commands.hex",commands);$readmemh("values.hex",values);
        $readmemh("bc.hex",bc);$readmemh("states.hex",states);$readmemh("outputs.hex",outputs);
        fd=$fopen("rtl_results.csv","w");$fdisplay(fd,"kind,token,index,wide,packed,cycle");
        clear_core();
        if(RANK==1) begin
            first_pair();@(negedge clk);
            if(stage!==11 || dut.rotation_valid!==1) begin
                $display("MISMATCH ROMOUT reset missed boundary");errors=errors+1;
            end
            reset=1;repeat(3) @(negedge clk);reset=0;
            while(!start_ready) @(negedge clk);
            $display("RESULT ROMOUT reset_inflight=checked");
        end
        if(RANK==1) begin
            first_pair();
            if(stage!==12 || dut.rotation_valid!==1) begin
                $display("MISMATCH PHASE reset missed captured-ROM boundary");errors=errors+1;
            end
            reset=1;repeat(3) @(negedge clk);reset=0;
            while(!start_ready) @(negedge clk);
            $display("RESULT PHASE reset_inflight=checked");
        end
        // Abort while the new product register is valid, before its add/write.
        // The original in-flight recurrence reset test below is retained.
        if(RANK==1) begin
            {alpha,beta,gamma,skip}=commands[0];start_valid=1;
            @(negedge clk);start_valid=0;
            for(i=0;i<P*RANK;i=i+1) begin
                while(!v_ready) @(negedge clk);
                v=values[i];v_valid=1;@(negedge clk);
            end
            v_valid=0;while(!bc_ready) @(negedge clk);
            {qx,qy,kx,ky,increment}=bc[0];bc_valid=1;
            @(negedge clk);bc_valid=0;
            while(stage==12 || stage==11) @(negedge clk);
            if(stage!==10 || dut.rotation_valid!==1) begin
                $display("MISMATCH reset test missed rotation stage");errors=errors+1;
            end
            // Assert reset now, before the next active edge consumes products.
            reset=1;repeat(3) @(negedge clk);reset=0;
            while(!start_ready) @(negedge clk);
            $display("RESULT ROTATION reset_inflight=checked");
        end
        // Abort with the new K*V boundary valid before any state write.
        if(RANK==1) begin
            load_token(0,1);
            while(!dut.valid_pipe[2]) @(negedge clk);
            if(sw!==0) begin
                $display("MISMATCH KV reset test missed first product boundary");errors=errors+1;
            end
            clear_core();
            $display("RESULT KV reset_inflight=checked");
        end
        if(RANK==1) begin
            load_token(0,1);
            while(!dut.valid_pipe[5]) @(negedge clk);
            if(sw!==0) begin $display("MISMATCH SUM reset missed boundary");errors=errors+1;end
            clear_core();$display("RESULT SUM reset_inflight=checked");
        end
        if(RANK==1) begin
            load_token(0,1);
            while(!dut.valid_pipe[1]) @(negedge clk);
            if(sw!==0) begin $display("MISMATCH READ reset missed boundary");errors=errors+1;end
            clear_core();$display("RESULT READ reset_inflight=checked");
        end
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
        if(RANK==1) begin
            if(rotation_accepts!=TOKENS*N/4 || rotation_adds!=rotation_accepts ||
               rotation_writes!=2*rotation_accepts || tag_pending) begin
                $display("MISMATCH rotation counts");errors=errors+1;
            end
            $display("RESULT ROTATION accepts=%0d adds=%0d tagged_writes=%0d reset_clear_checks=%0d",
                     rotation_accepts,rotation_adds,rotation_writes,post_reset_checks);
        end
        if(RANK==1) begin
            if(kv_admits!=TOKENS*P*N || kv_consumes!=kv_admits || kv_tag_q_checks!=kv_admits) begin
                $display("MISMATCH KV stage counts");errors=errors+1;
            end
            $display("RESULT KV admits=%0d consumes=%0d tag_q_checks=%0d",kv_admits,kv_consumes,kv_tag_q_checks);
        end
        if(RANK==1) begin
            if(sum_admits!=TOKENS*P*N || sum_consumes!=sum_admits) begin
                $display("MISMATCH SUM stage counts");errors=errors+1;
            end
            $display("RESULT SUM admits=%0d consumes=%0d",sum_admits,sum_consumes);
        end
        if(RANK==1) begin
            if(phase_admits!=TOKENS*N/4 || phase_consumes!=phase_admits) begin
                $display("MISMATCH PHASE stage counts");errors=errors+1;
            end
            $display("RESULT PHASE admits=%0d consumes=%0d",phase_admits,phase_consumes);
        end
        if(RANK==1) begin
            if(rom_admits!=TOKENS*N/4 || rom_consumes!=rom_admits) begin
                $display("MISMATCH ROMOUT stage counts");errors=errors+1;
            end
            $display("RESULT ROMOUT admits=%0d consumes=%0d",rom_admits,rom_consumes);
        end
        if(RANK==1) begin
            if(read_admits!=TOKENS*P*N || read_consumes!=read_admits) begin
                $display("MISMATCH READ stage counts");errors=errors+1;
            end
            $display("RESULT READ admits=%0d consumes=%0d",read_admits,read_consumes);
            if($bits(dut.read_state)!=32 || $bits(dut.state_mem[0])!=16) begin
                $display("MISMATCH WIDEN storage contract");errors=errors+1;
            end
            $display("RESULT WIDEN conversions=%0d persistent_bits=16 read_bits=32",read_admits);
        end
        if(writes!=TOKENS*P*N || reads!=TOKENS*P*RANK) errors=errors+1;
        $display("RESULT rank=%0d tokens=%0d writes=%0d outputs=%0d errors=%0d bubbles=%0d output_stalls=%0d consecutive_write_pairs=%0d",RANK,TOKENS,writes,reads,errors,bubble_count,stall_count,consecutive);
        if(errors==0) $display("PASS");else $display("FAIL");
        $finish;
    end
    initial begin #20000000;$display("FAIL timeout");$finish;end
endmodule
