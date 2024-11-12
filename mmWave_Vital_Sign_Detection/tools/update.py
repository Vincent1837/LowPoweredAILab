from subprocess import check_call
import os
from glob import glob
import importlib.util


def main():
    root_dir = os.getcwd()
    python_dir = os.path.abspath(glob("tools\\python-3.8.*")[0])
    python = os.path.abspath(os.path.join(python_dir, "python.exe"))

    # Check that pip is installed
    check_call([python, "-m", "pip", "--version"])
    pip_install = [python, "-m", "pip", "install", "--no-warn-script-location"]

    # Install and import dulwich
    if importlib.util.find_spec("dulwich") is None:
        check_call(pip_install + ["urllib3", "certifi"])
        check_call(pip_install + ["dulwich", "--global-option=--pure"])

if __name__ == "__main__":
    main()
