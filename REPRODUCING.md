# Reproduce the Mamba-3 phase-precision investigation

The complete report is in research_report.html (standalone) and research_report.md.
Raw numerical/build evidence is in evidence/. Models and Python packages are not included.
The same recorded inputs needed by analyze.py/make_report.py are also placed in
source/results and source/hardware/output_files so reports can be regenerated
without downloading checkpoints once the numerical/report dependencies exist.

Create a Python 3.12 environment. Install torch==2.8.0 from the CUDA 12.8 wheel index,
then install the versions in requirements-lock.txt (torch's CUDA build may need its
separate index). NumPy, Numba, matplotlib, tokenizers, pyarrow, einops and Markdown
are the principal non-PyTorch packages. On a different GPU/OS, results may differ.

Run fetch_assets.py to download the pinned official model/code/tokenizer assets.
Compare every dataset asset against evidence/asset_manifest.json before evaluating;
the fetch script's dataset URL resolves main. It never needs Hugging Face credentials
for the public assets used here. torch.load uses weights_only=True.

Run validation.py first. Then run, for each model name:
  python experiment.py --name mamba3-siso-187m --device cuda:0 --stage main
  python experiment.py --name mamba3-siso-187m --device cuda:0 --stage replicate
  python experiment.py --name mamba3-siso-187m --device cuda:0 --stage long
  python retrieval.py --name mamba3-siso-187m --device cuda:0
  python state_study.py --name mamba3-siso-187m --device cuda:0
Repeat with mamba3-mimo-187m. Files are resumable by completed configuration.
analyze.py and make_report.py rebuild summaries/reports after results exist; make_report.py
also expects the hardware evidence. The scripts' default deliverable location is
two parent folders above the source folder, under outputs/phase_precision_results.

FPGA: the tested project is hardware/phase_lab.qpf for EP4CE115F29C7.
Run hardware_vectors.py first if regenerating ROMs; already tested ROMs are included.
Use Quartus Lite 18.1: quartus_sh --flow compile phase_lab from hardware/.
Simulate phase_rotator.v, phase_lab.v, tb_phase_lab.v with ModelSim (simulated
altsource_probe stand-in is defined in the testbench). Expected result: PASS,
1024 vectors, 1028 cycles, zero errors for both cores.

The included SOF is the exact volatile test image, not a full-model accelerator.
Before programming another board, verify its device, pin map, and JTAG chain.
run_board_test.tcl intentionally requires one device with ID 0x020F70DD and the
8-source-bit/192-probe-bit ELAB node. Quartus 18.1 exposes only the low four
characters of the PHASELAB instance label. Both cores compare against their own
quantized arithmetic. Passing the board test does not mean the 8-bit model is accurate.

Upstream source files retain the original Apache-2.0 license. Vendor IP is generated
by the user's installed Quartus tools and is governed by their applicable licenses.
This bundle does not redistribute Quartus, Python dependencies, or model weights.
