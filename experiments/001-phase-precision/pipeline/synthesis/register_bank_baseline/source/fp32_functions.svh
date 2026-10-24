// Portable finite-width arithmetic for the research datapath. RNE, gradual
// underflow, infinity and canonical quiet NaN. Separate multiply/add, no FMA.
// Functions are combinational; the caller supplies pipeline registers.
function automatic [4:0] leading24(input [23:0] x);
    integer bit_index;
    begin
        leading24=24;
        for(bit_index=0;bit_index<24;bit_index=bit_index+1)
            if(x[bit_index]) leading24=23-bit_index;
    end
endfunction

function automatic [4:0] leading27(input [26:0] x);
    integer bit_index;
    begin
        leading27=27;
        for(bit_index=0;bit_index<27;bit_index=bit_index+1)
            if(x[bit_index]) leading27=26-bit_index;
    end
endfunction

function automatic [63:0] rne_shift(input [63:0] x, input integer shift);
    reg [63:0] q, mask, half, remn;
    begin
        if (shift <= 0) rne_shift = x << (-shift);
        else if (shift > 63) rne_shift = 0;
        else begin
            q = x >> shift;
            mask = (64'h1 << shift)-1;
            half = 64'h1 << (shift-1);
            remn = x & mask;
            rne_shift = q + ((remn > half) || ((remn == half) && q[0]));
        end
    end
endfunction

function automatic [27:0] shift_jam(input [27:0] x, input integer shift);
    reg [27:0] mask;
    begin
        if (shift == 0) shift_jam = x;
        else if (shift >= 28) shift_jam = {27'b0,|x};
        else begin
            mask = (28'h1 << shift)-1;
            shift_jam = (x >> shift) | ((x & mask) != 0);
        end
    end
endfunction

function automatic [31:0] fp_add(input [31:0] a, input [31:0] b);
    integer ea, eb, et, left;
    reg sa, sb, st;
    reg [23:0] ma, mb, mt;
    reg [27:0] xa, xb, total;
    reg [24:0] rounded;
    reg [7:0] encoded;
    begin
        ea = (a[30:23] == 0) ? 1 : a[30:23];
        eb = (b[30:23] == 0) ? 1 : b[30:23];
        ma = {a[30:23] != 0,a[22:0]}; mb = {b[30:23] != 0,b[22:0]};
        sa = a[31]; sb = b[31];
        if ((ea < eb) || ((ea == eb) && (ma < mb))) begin
            et=ea;ea=eb;eb=et;mt=ma;ma=mb;mb=mt;st=sa;sa=sb;sb=st;
        end
        xa = {1'b0,ma,3'b0}; xb = shift_jam({1'b0,mb,3'b0},ea-eb);
        total = (sa == sb) ? xa+xb : xa-xb;
        if (total[27]) begin total=shift_jam(total,1);ea=ea+1;end
        left=leading27(total[26:0]);
        if(left>ea-1) left=ea-1;
        total=total<<left;ea=ea-left;
        rounded = {1'b0,total[26:3]} + (total[2] && (total[1] || total[0] || total[3]));
        if (rounded[24]) begin rounded=rounded>>1;ea=ea+1;end
        encoded = ((ea == 1) && !rounded[23]) ? 0 : ea;
        if ((a[30:23] == 255 && a[22:0] != 0) ||
            (b[30:23] == 255 && b[22:0] != 0) ||
            (a[30:0] == 31'h7f800000 && b[30:0] == 31'h7f800000 && a[31] != b[31]))
            fp_add = 32'h7fc00000;
        else if (a[30:23] == 255) fp_add = a;
        else if (b[30:23] == 255) fp_add = b;
        else if (total == 0) fp_add = {(a[31] && b[31]),31'b0};
        else if (ea >= 255) fp_add = {sa,8'hff,23'b0};
        else fp_add = {sa,encoded,rounded[22:0]};
    end
endfunction

function automatic [31:0] fp_mul(input [31:0] a, input [31:0] b);
    integer ea, eb, e, shift;
    reg [4:0] za,zb;
    reg [23:0] ma, mb;
    reg [47:0] product;
    reg [63:0] rounded;
    reg sign_bit;
    reg [7:0] encoded;
    begin
        sign_bit=a[31]^b[31];
        ea=(a[30:23]==0)?1:a[30:23];eb=(b[30:23]==0)?1:b[30:23];
        ma={a[30:23]!=0,a[22:0]};mb={b[30:23]!=0,b[22:0]};
        za=leading24(ma);zb=leading24(mb);
        ma=ma<<za;ea=ea-za;mb=mb<<zb;eb=eb-zb;
        product=ma*mb;e=ea+eb-127;
        shift=23;
        if (product[47]) begin shift=24;e=e+1;end
        if (e<=0) shift=shift+1-e;
        rounded=rne_shift({16'b0,product},shift);
        if (e<=0) e=1;
        if (rounded[24]) begin rounded=rounded>>1;e=e+1;end
        encoded=((e==1) && !rounded[23])?0:e;
        if ((a[30:23]==255 && a[22:0]!=0) || (b[30:23]==255 && b[22:0]!=0) ||
            ((a[30:23]==255 && b[30:0]==0) || (b[30:23]==255 && a[30:0]==0)))
            fp_mul=32'h7fc00000;
        else if (a[30:23]==255 || b[30:23]==255) fp_mul={sign_bit,8'hff,23'b0};
        else if (a[30:0]==0 || b[30:0]==0) fp_mul={sign_bit,31'b0};
        else if (e>=255) fp_mul={sign_bit,8'hff,23'b0};
        else fp_mul={sign_bit,encoded,rounded[22:0]};
    end
endfunction

function automatic [31:0] half_to_float(input [15:0] h);
    integer e;
    reg [4:0] zeros;
    reg [10:0] m;
    reg [7:0] encoded;
    begin
        m={h[14:10]!=0,h[9:0]};e=(h[14:10]==0)?-14:(h[14:10]-15);
        zeros=leading24({m,13'b0});m=m<<zeros;e=e-zeros;
        encoded=e+127;
        if (h[14:0]==0) half_to_float={h[15],31'b0};
        else if (h[14:10]==31) half_to_float={h[15],8'hff,h[9:0],13'b0};
        else half_to_float={h[15],encoded,m[9:0],13'b0};
    end
endfunction

function automatic [15:0] float_to_half(input [31:0] a);
    integer e,shift;
    reg [23:0] m;
    reg [63:0] rounded;
    reg [4:0] encoded;
    begin
        e=a[30:23]-127+15;m={a[30:23]!=0,a[22:0]};
        shift=13;
        if (e<=0) shift=shift+1-e;
        rounded=rne_shift({40'b0,m},shift);
        if (e<=0) e=1;
        if (rounded[11]) begin rounded=rounded>>1;e=e+1;end
        encoded=((e==1) && !rounded[10])?0:e;
        if (a[30:23]==255) float_to_half={a[31],5'h1f,(a[22:0]!=0)?10'h200:10'b0};
        else if (e>=31) float_to_half={a[31],5'h1f,10'b0};
        else float_to_half={a[31],encoded,rounded[9:0]};
    end
endfunction
