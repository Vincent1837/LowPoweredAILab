from subprocess import check_call, CalledProcessError
import os
from glob import glob


def main():
    python_dir = os.path.abspath(glob("tools\\python-3.8.*")[0])
    os.chdir(python_dir)

    with open("python38._pth", "r") as f:
        contents = f.read()

    if "#import site" in contents:
        with open("python38._pth", "w") as f:
            f.write(contents.replace("#import site", "import site"))

    try:
        check_call([".\\python.exe", "-m", "pip", "--version"])
    except CalledProcessError:
        check_call([".\\python.exe", "..\\get-pip.py", "--no-warn-script-location"])


if __name__ == "__main__":
    main()
