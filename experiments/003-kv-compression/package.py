"""Package the current experiment; large external assets stay excluded."""
import subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(R.parents[1]/"tools/package_experiment.py"),str(R)],check=True)
