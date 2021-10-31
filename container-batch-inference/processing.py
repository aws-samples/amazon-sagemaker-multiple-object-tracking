import subprocess
import sys

dcn_cmd = "cd /DCNv2; sh make.sh"
subprocess.run(dcn_cmd, shell=True)
subprocess.run("python3 predict.py", shell=True)