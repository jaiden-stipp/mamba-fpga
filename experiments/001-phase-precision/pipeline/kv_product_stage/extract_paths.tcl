# Read completed routed databases with their original constraints.
# quartus_sta -t extract_paths.tcl <project> <output-prefix> <baseline|staged>
load_package sta
project_open -revision pipeline [lindex $argv 0]
create_timing_netlist -model slow
read_sdc
update_timing_netlist
set prefix [lindex $argv 1]
if {[lindex $argv 2] eq "baseline"} {
    report_timing -setup -from [get_registers {*k_ram*}] \
        -to [get_registers {now1*}] -npaths 1 -detail full_path \
        -file "${prefix}-kram-to-now1.rpt"
} else {
    report_timing -setup -from [get_registers {*k_ram*}] \
        -to [get_registers {kv_now_product*}] -npaths 1 -detail full_path \
        -file "${prefix}-kram-to-product.rpt"
    report_timing -setup -from [get_registers {kv_now_product*}] \
        -to [get_registers {now1*}] -npaths 1 -detail full_path \
        -file "${prefix}-product-to-now1.rpt"
}
delete_timing_netlist
project_close
