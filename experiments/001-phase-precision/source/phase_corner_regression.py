"""Seed-locked E001 RTL regression; Python standard library + ModelSim only.

Run from any directory. Generated evidence is separate from the original board
ROMs/build. Compiled simulator files live in a temporary directory outside E001.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from phase_reference import (coefficients, pack_golden, pack_vector,
                             quantized_rotation, round_even_ratio, signed,
                             unpack_vector)

ROOT = Path(__file__).resolve().parents[1]
HW = ROOT / 'source/hardware'
SEED = 0xE0014096
NEW_COUNT = 4096
TICKS = 1 << 24
CATEGORIES = ('phase_wraparound', 'signed_extrema', 'zero_increments',
              'rounding_boundaries')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_text(path, content):
    path.write_text(content, encoding='utf-8', newline='\n')


def write_hex(path, words, digits):
    write_text(path, ''.join(f'{word:0{digits}x}\n' for word in words))


def read_hex(path):
    return [int(line, 16) for line in path.read_text().splitlines() if line.strip()]


class LCG32:
    """Specified integer PRNG; independent of Python's random implementation."""
    def __init__(self, seed):
        self.state = seed

    def next(self):
        self.state = (1664525 * self.state + 1013904223) & 0xffffffff
        return self.state

    def coordinate(self):
        return signed(self.next() >> 16, 16)


def legacy_reference(cosine, sine):
    vectors = read_hex(HW / 'vectors.hex')
    expected = read_hex(HW / 'golden.hex')
    assert len(vectors) == len(expected) == 1024
    p16 = p8 = 0
    states = []
    for word, golden in zip(vectors, expected):
        s16, s8, x, y = unpack_vector(word)
        guard = quantized_rotation(p16, s16, x, y, 16, cosine, sine)
        low = quantized_rotation(p8, s8, x, y, 8, cosine, sine)
        assert pack_golden(guard, low) == golden, 'Legacy golden reference changed'
        p16, p8 = guard[0], low[0]
        states.append((guard, low))
    return vectors, expected, states


def generate(cosine, sine):
    legacy = set(read_hex(HW / 'vectors.hex'))
    vectors, golden, states = [], [], []
    seen = set(legacy)
    rng = LCG32(SEED)
    # Run new cases from reset, separately from the original ROM. Both cores
    # and the high-precision model start at zero; no unavailable legacy angles
    # or inherited low-precision phase offset enter the precision comparison.
    p16 = p8 = hp_phase = 0
    initial_phase = {'guard': 0, 'low': 0, 'higher_precision_ticks': 0}
    records = []
    for category in CATEGORIES:
        for local in range(1024):
            before16, before8 = p16, p8
            if category == 'phase_wraparound':
                radius = (local // 2) % 64
                target = 65535 - radius if local % 2 == 0 else radius
                step = signed(target - p16, 16)
            elif category == 'signed_extrema':
                directed = (-32768, 32767, -1, 1, 0, -32767, 32766, 2)
                step = directed[local] if local < len(directed) else signed(rng.next() >> 16, 16)
            elif category == 'zero_increments':
                step = 0
            else:
                # All 256 LUT bins: just below, exactly at, just above each
                # halfway point; 255 also exercises a carry into the next bin.
                target = ((local // 4) << 8) | (127, 128, 129, 255)[local % 4]
                step = signed(target - p16, 16)
            jitter = 0 if category == 'zero_increments' else (-127, 0, 127)[local % 3]
            increment_ticks = step * 256 + jitter
            s16 = round_even_ratio(increment_ticks, 256) & 65535
            s8 = round_even_ratio(increment_ticks, 65536) & 255
            for attempt in range(1000):
                if category == 'signed_extrema':
                    extreme = -32768 if local % 2 == 0 else 32767
                    x, y = (extreme, rng.coordinate()) if local % 4 < 2 else (rng.coordinate(), extreme)
                    if local < 4 and attempt == 0:
                        x, y = ((-32768, -32768), (-32768, 32767),
                                (32767, -32768), (32767, 32767))[local]
                elif category == 'zero_increments' and local < 9 and attempt == 0:
                    x, y = ((0, 0), (0, -32768), (-32768, 0), (0, 32767),
                            (32767, 0), (-1, 1), (1, -1),
                            (-32768, -32768), (32767, 32767))[local]
                else:
                    x, y = rng.coordinate(), rng.coordinate()
                word = pack_vector(s16, s8, x, y)
                if word not in seen:
                    break
            else:
                raise AssertionError('Cannot construct a unique vector')
            seen.add(word)
            guard = quantized_rotation(p16, s16, x, y, 16, cosine, sine)
            low = quantized_rotation(p8, s8, x, y, 8, cosine, sine)
            p16, p8 = guard[0], low[0]
            hp_phase = (hp_phase + increment_ticks) % TICKS
            theta = math.tau * hp_phase / TICKS
            hp_real = (x * math.cos(theta) - y * math.sin(theta)) / 1024
            hp_imag = (x * math.sin(theta) + y * math.cos(theta)) / 1024
            records.append({
                'index': len(vectors), 'new_index': len(records), 'category': category,
                'vector_hex': f'{word:014x}', 'increment_ticks_2p24_per_turn': increment_ticks,
                'step16': s16, 'step8': s8, 'x_q5_10': x, 'y_q5_10': y,
                'phase16_before': before16, 'phase8_before': before8,
                'phase16': p16, 'phase8': p8, 'higher_precision_phase_ticks': hp_phase,
                'higher_precision_real': hp_real, 'higher_precision_imag': hp_imag,
            })
            vectors.append(word)
            golden.append(pack_golden(guard, low))
            states.append((guard, low))
    new_words = vectors
    assert len(new_words) == len(set(new_words)) == NEW_COUNT
    assert not (set(new_words) & legacy), 'A new vector overlaps the original ROM'
    rounding = [r for r in records if r['category'] == 'rounding_boundaries']
    assert {(r['phase16'] >> 8, r['phase16'] & 255) for r in rounding} == {
        (b, fraction) for b in range(256) for fraction in (127, 128, 129, 255)}
    zeros = [r for r in records if r['category'] == 'zero_increments']
    assert all(r['step16'] == r['step8'] == r['increment_ticks_2p24_per_turn'] == 0
               and r['phase16'] == r['phase16_before'] and r['phase8'] == r['phase8_before']
               for r in zeros)
    wraps = [r for r in records if r['category'] == 'phase_wraparound']
    positive_wraps = sum(r['phase16_before'] + signed(r['step16'], 16) > 65535 for r in wraps)
    negative_wraps = sum(r['phase16_before'] + signed(r['step16'], 16) < 0 for r in wraps)
    assert positive_wraps >= 500 and negative_wraps >= 500
    coverage = {
        'new_vectors_per_category': {c: sum(r['category'] == c for r in records) for c in CATEGORIES},
        'unique_new_vectors': len(set(new_words)), 'overlap_with_legacy_vectors': 0,
        'positive_phase_wraps_in_wrap_category': positive_wraps,
        'negative_phase_wraps_in_wrap_category': negative_wraps,
        'rounding_lut_bins': 256, 'rounding_fractions_per_bin': [127, 128, 129, 255],
        'even_bin_halfway_ties': 128, 'odd_bin_halfway_ties': 128,
        'rounding_address_wrap_cases': sum(r['phase16'] >= 65408 for r in rounding),
        'zero_increment_phase_hold_checks_per_core': len(zeros),
        'signed_min_x_cases': sum(r['x_q5_10'] == -32768 for r in records),
        'signed_max_x_cases': sum(r['x_q5_10'] == 32767 for r in records),
        'signed_min_y_cases': sum(r['y_q5_10'] == -32768 for r in records),
        'signed_max_y_cases': sum(r['y_q5_10'] == 32767 for r in records),
        'signed_min_increment16_cases': sum(r['step16'] == 32768 for r in records),
        'signed_max_increment16_cases': sum(r['step16'] == 32767 for r in records),
    }
    return vectors, golden, states, records, initial_phase, coverage


def tool_path(name, tool_dir):
    path = tool_dir / (name + '.exe') if tool_dir else shutil.which(name)
    if not path or not Path(path).is_file():
        raise RuntimeError(f'{name} unavailable; pass --modelsim-bin')
    return str(path)


def run_tool(command, cwd):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=120)
    return result.returncode, result.stdout + result.stderr


def clean_transcript(text):
    # Remove wall-clock dates, paths, and run banners; retain simulator evidence.
    kept = [line for line in text.splitlines() if re.match(
        r'^# (?:RESULT |PASS$|FAIL|\*\* (?:Fatal|Error|Warning)|Errors:)', line)]
    return '\n'.join(kept) + '\n'


def simulate(stage, vectors, golden, tools, name, output):
    write_hex(stage / 'vectors.hex', vectors, 14)
    write_hex(stage / 'golden.hex', golden, 29)
    command = [tools['vsim'], '-c', '-onfinish', 'exit', '-lib', 'simlib',
               'tb_phase_lab', f'-gVECTOR_COUNT={len(vectors)}',
               '+RTL_RESULTS=rtl_results.csv', '-do',
               'onerror {quit -code 1}; run -all; quit -code 0']
    code, transcript = run_tool(command, stage)
    write_text(output / f'{name}_transcript.log', clean_transcript(transcript))
    rows = []
    if (stage / 'rtl_results.csv').is_file():
        with (stage / 'rtl_results.csv').open(newline='') as handle:
            rows = [{key: int(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    matches = re.findall(r'RESULT completed=(\d+) guard_errors=(\d+) low_errors=(\d+) '
                         r'cycles=(\d+) phase_guard=(\d+) phase_low=(\d+)', transcript)
    status = dict(zip(('completed', 'guard_errors', 'low_errors', 'cycles',
                      'phase_guard', 'phase_low'), map(int, matches[-1]))) if matches else {}
    passed = code == 0 and bool(re.search(r'^# PASS\s*$', transcript, re.M)) and '# FAIL' not in transcript
    if not status:
        raise RuntimeError(f'{name}: simulator did not report a result\n{transcript[-6000:]}')
    return rows, status, passed, code


def analyze(rows, states, records, output):
    assert len(rows) == len(states), 'Missing or extra RTL outputs'
    comparison = []
    quantized_max = {'guard': 0, 'low': 0}
    mismatches = {'guard': 0, 'low': 0}
    metrics = {}
    for index, (row, (guard, low)) in enumerate(zip(rows, states)):
        assert row['index'] == index, 'RTL outputs out of order'
        item = dict(row)
        item['category'] = records[index]['category'] if records else 'legacy'
        for core, state in (('guard', guard), ('low', low)):
            _, address, real, imag = state
            equal = row[f'{core}_real'] == real and row[f'{core}_imag'] == imag and row[f'{core}_address'] == address
            item[f'{core}_status'] = 'PASS' if equal else 'FAIL'
            item[f'{core}_expected_real'] = real
            item[f'{core}_expected_imag'] = imag
            item[f'{core}_expected_address'] = address
            item[f'{core}_max_integer_error'] = max(abs(row[f'{core}_real'] - real),
                                                    abs(row[f'{core}_imag'] - imag))
            quantized_max[core] = max(quantized_max[core], item[f'{core}_max_integer_error'])
            mismatches[core] += not equal
            if records:
                record = records[index]
                er = row[f'{core}_real'] / (127 * 1024) - record['higher_precision_real']
                ei = row[f'{core}_imag'] / (127 * 1024) - record['higher_precision_imag']
                item[f'{core}_higher_precision_real_error'] = er
                item[f'{core}_higher_precision_imag_error'] = ei
                for group in ('all_new', record['category']):
                    metric = metrics.setdefault(core, {}).setdefault(group, {
                        'vectors': 0, 'max_abs_component_error': -1.0,
                        'max_component_error_vector_index': None,
                        'max_l2_vector_error': 0.0, 'sum_squared_component_error': 0.0})
                    metric['vectors'] += 1
                    component = max(abs(er), abs(ei))
                    if component > metric['max_abs_component_error']:
                        metric['max_abs_component_error'] = component
                        metric['max_component_error_vector_index'] = index
                    metric['max_l2_vector_error'] = max(metric['max_l2_vector_error'], math.hypot(er, ei))
                    metric['sum_squared_component_error'] += er * er + ei * ei
        comparison.append(item)
    for groups in metrics.values():
        for metric in groups.values():
            metric['component_rmse'] = math.sqrt(metric.pop('sum_squared_component_error') / (2 * metric['vectors']))
    fields = list(comparison[-1])
    filename = 'per_vector_comparison.csv' if records else 'legacy_per_vector_comparison.csv'
    with (output / filename).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(comparison)
    return mismatches, quantized_max, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modelsim-bin', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'evidence/phase_corner_regression')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        execute(args, output)
    except Exception as error:
        message = f'FAIL seed=0x{SEED:08x} {type(error).__name__}: {error}\n'
        write_text(output / 'PASS_FAIL.log', message)
        write_text(output / 'summary.json', json.dumps({
            'status': 'FAIL', 'seed': SEED, 'error': str(error)}, indent=2) + '\n')
        raise


def execute(args, output):
    # Never leave an old PASS report after a failed rerun.
    write_text(output / 'PASS_FAIL.log', f'RUNNING seed=0x{SEED:08x}\n')
    write_text(output / 'summary.json', json.dumps({'status': 'RUNNING', 'seed': SEED}, indent=2) + '\n')
    cosine, sine = coefficients()
    assert [signed(x, 8) for x in read_hex(HW / 'cos.hex')] == cosine
    assert [signed(x, 8) for x in read_hex(HW / 'sin.hex')] == sine
    legacy_vectors, legacy_golden, legacy_states = legacy_reference(cosine, sine)
    vectors, golden, states, records, initial_phase, coverage = generate(cosine, sine)
    # Re-generation is independent of simulator output and must be identical.
    assert (vectors, golden, records) == (lambda r: (r[0], r[1], r[3]))(generate(cosine, sine))
    write_hex(output / 'vectors.hex', vectors, 14)
    write_hex(output / 'golden.hex', golden, 29)
    write_text(output / 'new_vectors.jsonl', ''.join(json.dumps(r, sort_keys=True) + '\n' for r in records))
    tools = {name: tool_path(name, args.modelsim_bin) for name in ('vlib', 'vlog', 'vsim')}
    code, version = run_tool([tools['vsim'], '-version'], output)
    assert code == 0, version
    with tempfile.TemporaryDirectory(prefix='e001-phase-regression-') as temp:
        stage = Path(temp)
        for name in ('cos.hex', 'sin.hex'):
            shutil.copyfile(HW / name, stage / name)
        code, log = run_tool([tools['vlib'], 'simlib'], stage)
        assert code == 0, log
        code, log = run_tool([tools['vlog'], '-sv', '-work', 'simlib'] + [str(HW / name) for name in (
            'phase_rotator.v', 'phase_lab.v', 'tb_phase_lab.v')], stage)
        assert code == 0 and not re.search(r'\*\* (?:Error|Warning)', log), log
        write_text(output / 'compile.log', '\n'.join(line for line in log.splitlines() if
                   line.startswith('-- Compiling') or line.startswith('Errors:')) + '\n')
        legacy_rows, legacy_status, legacy_pass, _ = simulate(stage, legacy_vectors, legacy_golden, tools, 'legacy', output)
        assert legacy_pass and len(legacy_rows) == 1024 and legacy_status['cycles'] == 1028, legacy_status
        assert legacy_status['phase_guard'] == legacy_states[-1][0][0] and legacy_status['phase_low'] == legacy_states[-1][1][0]
        legacy_mismatches, _, _ = analyze(legacy_rows, legacy_states, [], output)
        assert all(value == 0 for value in legacy_mismatches.values()), legacy_mismatches
        rows, status, passed, exit_code = simulate(stage, vectors, golden, tools, 'extended', output)
        mismatches, quantized_max, metrics = analyze(rows, states, records, output)
        assert passed and status['cycles'] == len(vectors) + 4, status
        assert status['completed'] == len(vectors) and status['guard_errors'] == status['low_errors'] == 0
        assert status['phase_guard'] == states[-1][0][0] and status['phase_low'] == states[-1][1][0]
        assert all(value == 0 for value in mismatches.values()), mismatches
        # Exercise both RTL comparators and the simulator failure path.
        corrupted = list(golden)
        corrupted[0] ^= (1 << 0) | (1 << 50)
        _, negative, negative_pass, negative_code = simulate(stage, vectors, corrupted, tools, 'negative_control', output)
        # This ModelSim edition can exit zero on $fatal with -onfinish exit.
        # Require an explicit PASS plus clean counters for real test runs;
        # never interpret the simulator process return code alone as success.
        assert not negative_pass and negative['guard_errors'] == negative['low_errors'] == 1, negative
        assert 'FAIL' in (output / 'negative_control_transcript.log').read_text()
    source_paths = [HW / name for name in ('phase_rotator.v', 'phase_lab.v', 'tb_phase_lab.v',
                                         'cos.hex', 'sin.hex', 'vectors.hex', 'golden.hex')]
    source_paths += [ROOT / 'source' / name for name in ('phase_reference.py', 'phase_corner_regression.py', 'hardware_vectors.py')]
    artifact_names = ('vectors.hex', 'golden.hex', 'new_vectors.jsonl', 'per_vector_comparison.csv', 'legacy_per_vector_comparison.csv',
                      'compile.log', 'legacy_transcript.log', 'extended_transcript.log', 'negative_control_transcript.log')
    summary = {
        'status': 'PASS', 'evidence_type': 'RTL_simulation_not_new_board_measurement',
        'seed': SEED, 'seed_hex': f'0x{SEED:08x}',
        'prng': 'LCG32: state=(1664525*state+1013904223) mod 2^32; upper 16 bits for coordinates',
        'legacy_vectors': 1024, 'new_vectors': NEW_COUNT, 'total_vectors': 1024 + len(vectors),
        'checked_core_results': 2 * (1024 + len(vectors)), 'simulator_version': version.strip(),
        'legacy_simulation': legacy_status, 'extended_simulation': status, 'simulator_exit_code': exit_code,
        'negative_control': {'status': 'PASS_expected_RTL_failure_detected',
                             'corrupted_index': 0, 'corrupted_golden_bits': [0, 50],
                             'simulator_exit_code': negative_code, 'observed': negative},
        'coverage': coverage, 'legacy_quantized_reference_mismatches': legacy_mismatches,
        'quantized_reference_mismatches': mismatches,
        'max_quantized_reference_integer_error': quantized_max,
        'higher_precision_reference': {
            'scope': 'Phase-rotation block only, not model logits/perplexity or full Mamba-3',
            'vectors': NEW_COUNT, 'initial_state_after_reset': initial_phase,
            'phase': 'Exact modular integer accumulation of signed increments at 2^24 ticks/turn; float64 sin/cos',
            'inputs': 'Same exact Q5.10 inputs as RTL; no additional input-quantization error',
            'output_units': 'Input coordinate units: RTL signed product sums divided by 127*1024',
            'error_includes': 'Increment quantization, accumulation differences, address rounding, and 8-bit trig coefficients',
            'limitation': 'Synthetic block-level stress sequence only. Legacy raw floating increments are unavailable, so higher-precision errors are reported only for the separately reset new sequence.',
        },
        'higher_precision_errors': metrics,
        'source_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in source_paths},
        'artifact_sha256': {name: digest(output / name) for name in artifact_names},
    }
    write_text(output / 'summary.json', json.dumps(summary, indent=2) + '\n')
    lines = [f'PASS seed=0x{SEED:08x}', 'Scope: RTL simulation; no new FPGA board run.',
             'PASS legacy=1024 new=4096 total=5120 unique_new=4096 legacy_overlap=0',
             'PASS regenerated seed-locked vectors and golden data are identical',
             'PASS original 1024 golden words reproduced by shared integer reference',
             f'PASS checked_core_results={2 * (1024 + len(vectors))} legacy_cycles={legacy_status["cycles"]} new_cycles={status["cycles"]}',
             'PASS quantized reference: guard mismatches=0 max_integer_error=0; low mismatches=0 max_integer_error=0']
    lines += [f'PASS coverage {c}={coverage["new_vectors_per_category"][c]}' for c in CATEGORIES]
    lines += [f'PASS phase wraps: positive={coverage["positive_phase_wraps_in_wrap_category"]} negative={coverage["negative_phase_wraps_in_wrap_category"]}',
              'PASS rounding: all 256 bins; fractions 127,128,129,255; even/odd ties and address wrap',
              'PASS negative control: one corrupt expected real result per core detected as RTL FAIL']
    for core in ('guard', 'low'):
        m = metrics[core]['all_new']
        lines.append(f'HIGH_PRECISION {core}: max_abs_component_error={m["max_abs_component_error"]:.12g} '
                     f'max_l2_vector_error={m["max_l2_vector_error"]:.12g} component_rmse={m["component_rmse"]:.12g} '
                     f'worst_vector_index={m["max_component_error_vector_index"]}')
    lines += ['Higher-precision errors use input coordinate units; both cores and float64 rotation start from zero phase.',
              f'VECTORS_SHA256 {digest(output / "vectors.hex")}',
              f'GOLDEN_SHA256 {digest(output / "golden.hex")}',
              f'PER_VECTOR_SHA256 {digest(output / "per_vector_comparison.csv")}']
    write_text(output / 'PASS_FAIL.log', '\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'FAIL {type(error).__name__}: {error}')
        raise
