import os
import subprocess

script = os.path.join(os.path.expanduser("~"), ".ipython/profile_default/startup/start.sh")
subprocess.run(["bash", script])
