import sys
import subprocess

commands = ["-m pip install -r gesticulator/requirements.txt",
            "-m pip install -e .", # "." is the folder where setup.py is found
            "-m pip install -e gesticulator/visualization"]  
            #MJ: https://stackoverflow.com/questions/15031694/installing-python-packages-from-local-file-system-folder-to-virtualenv-with-pip:
            #  "-e" is optional here;   gesticulator/visualization is the top-level directory where "setup.py" is found
            #When you run pip install -e ., it does not install the package in the traditional sense.
            # Instead, it adds the current directory ('d:\\dropbox\\metaverse\\gesticulatorunity') 
            # to the Python system path (sys.path).
            # This allows you to work with the package's source code directly without the need for a separate installation.
            
            #When you install a package in editable mode using pip install -e ., the package's source directory is added to sys.path. This modification persists across Python sessions
            # and remains effective until you explicitly remove the package or uninstall it.

for cmd in commands:
    subprocess.check_call([sys.executable] + cmd.split())
