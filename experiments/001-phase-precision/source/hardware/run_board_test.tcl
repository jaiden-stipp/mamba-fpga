package require ::quartus::insystem_source_probe
set hardware [lindex [get_hardware_names] 0]
set devices [get_device_names -hardware_name $hardware]
if {[llength $devices]!=1} {error "Expected one device; found $devices"}
set device [lindex $devices 0]
if {![string match "*0x020F70DD*" $device]} {error "Unexpected device: $device"}
set infos [get_insystem_source_probe_instance_info -hardware_name $hardware -device_name $device]
puts "INSTANCES=$infos"
set found 0
foreach info $infos {
    # Quartus 18.1 exposes the low four characters of instance_id.
    if {[lindex $info 3]=="ELAB" && [lindex $info 1]==8 && [lindex $info 2]==192} {set index [lindex $info 0];set found 1}
}
if {!$found} {error "PHASELAB debug instance not found"}
start_insystem_source_probe -hardware_name $hardware -device_name $device
set f [open "board_runs.jsonl" w]
set failed 0
for {set run 0} {$run<10} {incr run} {
    write_source_data -instance_index $index -value_in_hex -value 01
    after 20
    set reset_value [string toupper [read_probe_data -instance_index $index -value_in_hex]]
    write_source_data -instance_index $index -value_in_hex -value 00
    after 20
    set value [string toupper [read_probe_data -instance_index $index -value_in_hex]]
    set value [format %048s $value]
    set magic [string range $value 0 7]
    scan [string range $value 8 15] %x cycles
    scan [string range $value 16 23] %x count
    scan [string range $value 24 31] %x errg
    scan [string range $value 32 39] %x errl
    scan [string range $value 40 43] %x phaseg
    scan [string range $value 44 45] %x phasel
    set marker [string range $value 46 47]
    set pass [expr {$magic=="50484153" && $marker=="A5" && $cycles==1028 && $count==1024 && $errg==0 && $errl==0 && $phaseg==40479 && $phasel==181}]
    if {!$pass} {set failed 1}
    set line "\{\"run\":$run,\"cycles\":$cycles,\"vectors\":$count,\"guard_errors\":$errg,\"low_errors\":$errl,\"phase_guard\":$phaseg,\"phase_low\":$phasel,\"pass\":$pass,\"probe_hex\":\"$value\"\}"
    puts $f $line;puts $line
}
close $f
end_insystem_source_probe
if {$failed} {error "Hardware self-test failed; see board_runs.jsonl"}
puts "BOARD_TEST_PASS: 10 runs, 1024 vectors per core per run"
