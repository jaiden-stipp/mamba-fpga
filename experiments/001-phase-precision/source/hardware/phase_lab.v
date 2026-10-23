// Default geometry/ROMs retain the original 1,024-vector board workload.
module phase_lab #(parameter VECTOR_COUNT=1024)(input wire CLOCK_50);
    wire [7:0] control;
    wire reset=control[0];
    wire [191:0] probe;
    altsource_probe #(.instance_id("PHASELAB"),.probe_width(192),.source_width(8),
        .source_initial_value("1"),.enable_metastability("YES")) debug_link(
        .probe(probe),.source(control),.source_clk(CLOCK_50),.source_ena(1'b1));
    localparam INDEX_BITS=(VECTOR_COUNT>1) ? $clog2(VECTOR_COUNT) : 1;
    reg [55:0] vectors[0:VECTOR_COUNT-1];
    reg [115:0] golden[0:VECTOR_COUNT-1];
    initial begin
        $readmemh("vectors.hex",vectors);
        $readmemh("golden.hex",golden);
    end
    reg [INDEX_BITS:0] issue;
    reg [55:0] data;
    reg [115:0] gold_0,gold_1,gold_2,gold_3;
    reg input_valid;
    wire valid_guard,valid_low;
    wire signed [24:0] rg,ig,rl,il;
    wire [7:0] ag,al,phase_low;
    wire [15:0] phase_guard;
    phase_rotator #(.ACC_BITS(16)) guarded(
        .clk(CLOCK_50),.reset(reset),.in_valid(input_valid),.increment(data[55:40]),
        .x_in(data[31:16]),.y_in(data[15:0]),.out_valid(valid_guard),
        .real_out(rg),.imag_out(ig),.address_out(ag),.phase(phase_guard));
    phase_rotator #(.ACC_BITS(8)) low_precision(
        .clk(CLOCK_50),.reset(reset),.in_valid(input_valid),.increment(data[39:32]),
        .x_in(data[31:16]),.y_in(data[15:0]),.out_valid(valid_low),
        .real_out(rl),.imag_out(il),.address_out(al),.phase(phase_low));
    reg [31:0] completed,errors_guard,errors_low,cycles;
    wire done=(completed==VECTOR_COUNT);
    wire [INDEX_BITS-1:0] read_index=(issue<VECTOR_COUNT) ? issue[INDEX_BITS-1:0] : {INDEX_BITS{1'b0}};
    always @(posedge CLOCK_50) begin
        data<=vectors[read_index];
        gold_0<=golden[read_index];
    end
    always @(posedge CLOCK_50) begin
        if (reset) begin
            issue<=0;gold_1<=0;gold_2<=0;gold_3<=0;
            input_valid<=0;completed<=0;errors_guard<=0;errors_low<=0;cycles<=0;
        end else begin
            if (!done) cycles<=cycles+1;
            input_valid<=(issue<VECTOR_COUNT);
            if (issue<VECTOR_COUNT) begin
                issue<=issue+1'b1;
            end
            gold_1<=gold_0;gold_2<=gold_1;gold_3<=gold_2;
            if (valid_guard) begin
                completed<=completed+1;
                if ((rg!==gold_3[24:0]) || (ig!==gold_3[49:25]) || (ag!==gold_3[107:100]))
                    errors_guard<=errors_guard+1;
                if ((rl!==gold_3[74:50]) || (il!==gold_3[99:75]) || (al!==gold_3[115:108]) || !valid_low)
                    errors_low<=errors_low+1;
            end
        end
    end
    assign probe={32'h50484153,cycles,completed,errors_guard,errors_low,phase_guard,phase_low,8'ha5};
endmodule
