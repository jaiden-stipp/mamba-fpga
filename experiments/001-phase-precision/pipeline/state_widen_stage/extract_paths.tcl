# Read an existing routed database with its original timing constraints.
# quartus_sta -t extract_paths.tcl <project> <report-file> <from-glob> <to-glob>
load_package sta
project_open -revision pipeline [lindex $argv 0]
create_timing_netlist -model slow
read_sdc
update_timing_netlist
report_timing -setup -from [get_registers [lindex $argv 2]] \
    -to [get_registers [lindex $argv 3]] -npaths 1 -detail full_path \
    -file [lindex $argv 1]
delete_timing_netlist
project_close
