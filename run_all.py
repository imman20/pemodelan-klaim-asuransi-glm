"""Jalankan unduhan, analisis, dan notebook dari root repo."""
import subprocess
import sys

for module in ["src.data.download", "src.pipeline", "src.notebook"]:
    subprocess.run([sys.executable, "-m", module], check=True)
