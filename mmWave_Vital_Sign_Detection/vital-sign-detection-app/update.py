from subprocess import check_call
import os
from glob import glob
import importlib.util


def main():
    root_dir = os.getcwd()
  
    repo_dir = "vital-sign-detection-app"
    python_dir = os.path.abspath(glob("tools\\python-3.8.*")[0])
    python = os.path.abspath(os.path.join(python_dir, "python.exe"))

    # Check that pip is installed
    check_call([python, "-m", "pip", "--version"])
    pip_install = [python, "-m", "pip", "install", "--no-warn-script-location"]

    # Install requirements and acconeer_utils
    os.chdir(repo_dir)
    check_call(pip_install + ["-r", "requirements.txt"])


if __name__ == "__main__":
    main()
