# E005 analytical results

These are outputs of an explicit-address model, not measured device traffic, latency or model quality. See [protocol](PROTOCOL.md).

| Requested schedule | Layout | Logical payload | Modeled total fetch | Relative to byte codes |
|---|---|---:|---:|---:|
| uniform4 | byte_codes | 16,384 B | 34,304 B | 1.000 |
| uniform4 | group_planes | 16,384 B | 34,304 B | 1.000 |
| uniform4 | tile4_planes | 16,384 B | 17,920 B | 0.522 |
| uniform4 | tile8_planes | 16,384 B | 17,920 B | 0.522 |
| uniform4 | tile16_planes | 16,384 B | 17,920 B | 0.522 |
| uniform4 | tile64_planes | 16,384 B | 17,920 B | 0.522 |
| uniform4 | global_planes | 16,384 B | 17,920 B | 0.522 |
| random26 | byte_codes | 16,384 B | 34,304 B | 1.000 |
| random26 | group_planes | 16,384 B | 34,304 B | 1.000 |
| random26 | tile4_planes | 16,384 B | 24,704 B | 0.720 |
| random26 | tile8_planes | 16,384 B | 26,112 B | 0.761 |
| random26 | tile16_planes | 16,384 B | 26,112 B | 0.761 |
| random26 | tile64_planes | 16,384 B | 26,112 B | 0.761 |
| random26 | global_planes | 16,384 B | 26,112 B | 0.761 |

147 configurations; 170 consistency checks passed. All layouts retain 32,768 payload bytes plus 1,536 metadata bytes.

A four-bit prefix does not imply half the physical bytes: the byte-code layout reads full bytes, and a group-major layout can fetch unused planes in the same unit. Grouping planes across groups helps uniform schedules. Scattered higher-depth groups can touch most units and erase the gain.

Next: replay genuine model-selected depth schedules, measure decoder/scheduler cost and hardware counters, and compare at matched answer quality and total memory. No eight-bit retention or arbitrary prefix is automatically a useful quantizer.
