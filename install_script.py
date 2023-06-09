import sys
import subprocess

commands = ["-m pip install -r gesticulator/requirements.txt",
            "-m pip install -e .", # "." is the folder where setup.py is found
            "-m pip install -e gesticulator/visualization"]  
            #https://stackoverflow.com/questions/15031694/installing-python-packages-from-local-file-system-folder-to-virtualenv-with-pip:
            #  "-e" is optional here;   gesticulator/visualization is the top-level directory where "setup.py" is found

for cmd in commands:
    subprocess.check_call([sys.executable] + cmd.split())
