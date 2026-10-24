// Resident single-head, P-channel tile. FP16 persistent state; FP32 arithmetic.
// Streaming order V: p then rank; BC: rank then pair. Host supplies projected
// rank-specific V, normalized/bias-adjusted B/C and nonlinear coefficients.
module mamba3_recurrent_tile #(
    parameter N=128, P=8, RANK=4, SPLIT_PAIR=(RANK>1), ANGLES=N/4
)(
    input wire clk, reset,
    input wire start_valid,
    output wire start_ready,
    input wire [31:0] alpha_in, beta_in, gamma_in, skip_in,
    input wire v_valid, output wire v_ready, input wire [31:0] v_in,
    input wire bc_valid, output wire bc_ready,
    input wire [31:0] qx_in,qy_in,kx_in,ky_in,
    input wire [15:0] increment_in,
    output wire y_valid, input wire y_ready,
    output wire [31:0] y_out, output wire [15:0] y_index,
    output reg done,
    output wire state_write_valid,
    output wire [31:0] state_write_index,
    output wire [15:0] state_write_data,
    output wire [31:0] state_write_wide,
    output wire [3:0] debug_stage
);
    `include "fp32_functions.svh"
    localparam CLEAR=0,IDLE=1,LOAD_V=2,LOAD_BC=3,WRITE_LO=4,WRITE_HI=5,
               RUN=6,DRAIN=7,COPY=8,OUTPUT_Y=9,ROTATE_ADD=10;
    reg [3:0] mode;
    assign debug_stage=mode;
    reg [31:0] alpha,beta,gamma,skip;
    reg [15:0] phase[0:ANGLES-1];
    reg [31:0] cosine[0:255],sine[0:255];
    initial begin
        $readmemh("cos_fp32.hex",cosine);$readmemh("sin_fp32.hex",sine);
    end
    (* ramstyle = "M9K" *) reg [15:0] state_mem[0:P*N-1];
    reg [31:0] v_mem[0:RANK-1][0:P-1],prev_v[0:RANK-1][0:P-1];
    reg [31:0] accum[0:RANK-1][0:P-1];
    integer count,vr,vp,br,bpair,issue,copy_index,out_index;
    integer r;
    reg [31:0] qx,qy,kx,ky;
    // One rank-1 boundary: rounded FP32 products -> original FP32 sums.
    // Only rotated pairs take this extra cycle. No arithmetic reassociation.
    reg [31:0] rot_qxc,rot_qyns,rot_qxs,rot_qyc;
    reg [31:0] rot_kxc,rot_kyns,rot_kxs,rot_kyc;
    reg rotation_valid;
    integer rotation_pair,rotation_rank;
    reg [7:0] rotation_address;
    wire [31:0] write_pair=rotation_valid?rotation_pair:bpair;
    wire [31:0] write_rank=rotation_valid?rotation_rank:br;
    wire [15:0] next_phase=(bpair<ANGLES)?(phase[bpair]+increment_in):16'b0;
    wire [15:0] selected_phase=(br==0)?next_phase:((bpair<ANGLES)?phase[bpair]:16'b0);
    wire round_up=(selected_phase[7:0]>128) || ((selected_phase[7:0]==128) && selected_phase[8]);
    wire [7:0] address=selected_phase[15:8]+round_up;
    wire [31:0] c=cosine[address],s=sine[address];
    reg [5:0] valid_pipe;
    integer indices[0:5];
    reg [15:0] stored0;
    wire [31:0] k0[0:RANK-1],pk0[0:RANK-1],q0[0:RANK-1];
    reg [31:0] v0[0:RANK-1],pv0[0:RANK-1];
    reg [31:0] q1[0:RANK-1],q2[0:RANK-1],q3[0:RANK-1];
    reg [31:0] old1,prev1,now1,old2,prev2,now2,updated3;
    reg [31:0] product4[0:RANK-1];
    reg [31:0] prev_dot,now_dot;
    integer j;
    always @* begin
        prev_dot=0;now_dot=0;
        for (j=0;j<RANK;j=j+1) begin
            prev_dot=fp_add(prev_dot,fp_mul(pk0[j],pv0[j]));
            now_dot=fp_add(now_dot,fp_mul(k0[j],v0[j]));
        end
    end
    assign start_ready=(mode==IDLE);
    assign v_ready=(mode==LOAD_V);
    assign bc_ready=(mode==LOAD_BC);
    assign y_valid=(mode==OUTPUT_Y);
    assign y_index=out_index;
    assign y_out=fp_add(accum[out_index%RANK][out_index/RANK],
                        fp_mul(skip,v_mem[out_index%RANK][out_index/RANK]));
    assign state_write_valid=valid_pipe[3];
    assign state_write_index=indices[3];
    assign state_write_data=float_to_half(updated3);
    assign state_write_wide=updated3;

    // Explicit synchronous, single-write RAM ports. A floating conversion
    // between RAM output and its register prevents useful RAM inference.
    always @(posedge clk) begin
        if(!reset && mode==CLEAR) state_mem[count]<=0;
        else if(!reset && valid_pipe[3]) state_mem[indices[3]]<=float_to_half(updated3);
        if(mode==RUN) stored0<=state_mem[issue];
    end
    wire [31:0] cache_read_index=(mode==COPY)?copy_index:issue%N;
    wire [31:0] bc_write_index=(mode==WRITE_LO)?
        (SPLIT_PAIR?write_pair:2*write_pair):(SPLIT_PAIR?(write_pair+N/2):(2*write_pair+1));
    wire cache_read_enable=(mode==RUN || mode==COPY) && cache_read_index<N;
    wire previous_write_enable=!reset && ((mode==CLEAR && count<N) || (mode==COPY && copy_index>0));
    wire [31:0] previous_write_index=(mode==CLEAR)?count:copy_index-1;
    genvar bank;
    generate for(bank=0;bank<RANK;bank=bank+1) begin: cache_bank
        wire bc_write_enable=!reset && write_rank==bank && (mode==WRITE_LO || mode==WRITE_HI);
        mamba3_cache_ram #(.DEPTH(N)) q_ram(.clk(clk),.write_enable(bc_write_enable),
            .write_address(bc_write_index),.write_data((mode==WRITE_LO)?qx:qy),
            .read_enable(cache_read_enable),.read_address(cache_read_index),.read_data(q0[bank]));
        mamba3_cache_ram #(.DEPTH(N)) k_ram(.clk(clk),.write_enable(bc_write_enable),
            .write_address(bc_write_index),.write_data((mode==WRITE_LO)?kx:ky),
            .read_enable(cache_read_enable),.read_address(cache_read_index),.read_data(k0[bank]));
        mamba3_cache_ram #(.DEPTH(N)) previous_ram(.clk(clk),.write_enable(previous_write_enable),
            .write_address(previous_write_index),.write_data((mode==CLEAR)?32'b0:k0[bank]),
            .read_enable(cache_read_enable),.read_address(cache_read_index),.read_data(pk0[bank]));
    end endgenerate

    always @(posedge clk) begin
        if (reset) begin
            mode<=CLEAR;count<=0;valid_pipe<=0;done<=0;out_index<=0;
            br<=0;bpair<=0;vr<=0;vp<=0;issue<=0;copy_index<=0;
            rotation_valid<=0;rotation_pair<=0;rotation_rank<=0;rotation_address<=0;
        end else begin
            done<=0;
            valid_pipe<={valid_pipe[4:0],(mode==RUN)};
            for (r=1;r<6;r=r+1) indices[r]<=indices[r-1];
            if (mode==RUN) begin
                indices[0]<=issue;
                for (r=0;r<RANK;r=r+1) begin
                    v0[r]<=v_mem[r][issue/N];pv0[r]<=prev_v[r][issue/N];
                end
            end
            if (valid_pipe[0]) begin
                old1<=fp_mul(alpha,half_to_float(stored0));prev1<=prev_dot;now1<=now_dot;
                for(r=0;r<RANK;r=r+1) q1[r]<=q0[r];
            end
            if (valid_pipe[1]) begin
                old2<=old1;prev2<=fp_mul(beta,prev1);now2<=fp_mul(gamma,now1);
                for(r=0;r<RANK;r=r+1) q2[r]<=q1[r];
            end
            if (valid_pipe[2]) begin
                updated3<=fp_add(fp_add(old2,prev2),now2);
                for(r=0;r<RANK;r=r+1) q3[r]<=q2[r];
            end
            if (valid_pipe[3]) begin
                for(r=0;r<RANK;r=r+1) product4[r]<=fp_mul(updated3,q3[r]);
            end
            if (valid_pipe[4]) begin
                for(r=0;r<RANK;r=r+1)
                    accum[r][indices[4]/N]<=fp_add(accum[r][indices[4]/N],product4[r]);
            end
            case(mode)
                CLEAR: begin
                    if(count<P) for(r=0;r<RANK;r=r+1) prev_v[r][count]<=0;
                    if(count<ANGLES) phase[count]<=0;
                    if(count==P*N-1) begin mode<=IDLE;count<=0;end else count<=count+1;
                end
                IDLE: if(start_valid) begin
                    alpha<=alpha_in;beta<=beta_in;gamma<=gamma_in;skip<=skip_in;
                    vr<=0;vp<=0;br<=0;bpair<=0;mode<=LOAD_V;
                end
                LOAD_V: if(v_valid) begin
                    v_mem[vr][vp]<=v_in;accum[vr][vp]<=0;
                    if(vr==RANK-1) begin
                        vr<=0;
                        if(vp==P-1) mode<=LOAD_BC;else vp<=vp+1;
                    end else vr<=vr+1;
                end
                LOAD_BC: if(bc_valid) begin
                    if(bpair<ANGLES) begin
                        if(br==0) phase[bpair]<=next_phase;
                        if(RANK==1) begin
                            rot_qxc<=fp_mul(qx_in,c);rot_qyns<=fp_mul(qy_in,{~s[31],s[30:0]});
                            rot_qxs<=fp_mul(qx_in,s);rot_qyc<=fp_mul(qy_in,c);
                            rot_kxc<=fp_mul(kx_in,c);rot_kyns<=fp_mul(ky_in,{~s[31],s[30:0]});
                            rot_kxs<=fp_mul(kx_in,s);rot_kyc<=fp_mul(ky_in,c);
                            rotation_pair<=bpair;rotation_rank<=br;rotation_address<=address;
                            rotation_valid<=1;
                        end else begin
                            qx<=fp_add(fp_mul(qx_in,c),fp_mul(qy_in,{~s[31],s[30:0]}));
                            qy<=fp_add(fp_mul(qx_in,s),fp_mul(qy_in,c));
                            kx<=fp_add(fp_mul(kx_in,c),fp_mul(ky_in,{~s[31],s[30:0]}));
                            ky<=fp_add(fp_mul(kx_in,s),fp_mul(ky_in,c));
                        end
                    end else begin qx<=qx_in;qy<=qy_in;kx<=kx_in;ky<=ky_in;end
                    mode<=(RANK==1 && bpair<ANGLES)?ROTATE_ADD:WRITE_LO;
                end
                ROTATE_ADD: if(rotation_valid) begin
                    qx<=fp_add(rot_qxc,rot_qyns);qy<=fp_add(rot_qxs,rot_qyc);
                    kx<=fp_add(rot_kxc,rot_kyns);ky<=fp_add(rot_kxs,rot_kyc);
                    mode<=WRITE_LO;
                end
                WRITE_LO: begin
                    mode<=WRITE_HI;
                end
                WRITE_HI: begin
                    rotation_valid<=0;
                    if(bpair==N/2-1) begin
                        bpair<=0;
                        if(br==RANK-1) begin issue<=0;mode<=RUN;end
                        else begin br<=br+1;mode<=LOAD_BC;end
                    end else begin bpair<=bpair+1;mode<=LOAD_BC;end
                end
                RUN: if(issue==P*N-1) mode<=DRAIN;else issue<=issue+1;
                DRAIN: if(valid_pipe==0) begin copy_index<=0;mode<=COPY;end
                COPY: begin
                    for(r=0;r<RANK;r=r+1) begin
                        if(copy_index<P) prev_v[r][copy_index]<=v_mem[r][copy_index];
                    end
                    if(copy_index==N) begin out_index<=0;mode<=OUTPUT_Y;end
                    else copy_index<=copy_index+1;
                end
                OUTPUT_Y: if(y_ready) begin
                    if(out_index==P*RANK-1) begin done<=1;mode<=IDLE;end
                    else out_index<=out_index+1;
                end
            endcase
        end
    end
endmodule
