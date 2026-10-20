create_clock -name CLOCK_50 -period 20.000 [get_ports {CLOCK_50}]
derive_clock_uncertainty
derive_pll_clocks
# No user data crosses physical I/O. The vendor debug bridge handles JTAG CDC.
