package require ::quartus::insystem_source_probe
set hardware [lindex [get_hardware_names] 0]
puts "HARDWARE=$hardware"
set devices [get_device_names -hardware_name $hardware]
puts "DEVICES=$devices"
