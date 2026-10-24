# Read a completed routed database; do not change constraints or build/program it.
# quartus_sta -t extract_paths.tcl <absolute project> <absolute output prefix> <baseline|staged>
load_package sta
project_open -revision pipeline [lindex $argv 0]
create_timing_netlist -model slow
read_sdc
update_timing_netlist
set prefix [lindex $argv 1]
set variant [lindex $argv 2]
if {$variant eq "baseline"} {
    report_timing -setup -from [get_registers {*phase_rtl_0*}] \
        -to [get_registers {ky*}] -npaths 1 -detail full_path \
        -file "${prefix}-phase-to-ky.rpt"
} else {
    report_timing -setup -from [get_registers {*phase_rtl_0*}] \
        -to [get_registers {rot_*}] -npaths 1 -detail full_path \
        -file "${prefix}-phase-to-products.rpt"
    report_timing -setup -from [get_registers {rot_k*}] \
        -to [get_registers {ky*}] -npaths 1 -detail full_path \
        -file "${prefix}-products-to-ky.rpt"
}
delete_timing_netlist
project_close
