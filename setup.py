import os
import subprocess


def install_requirements():
    requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_file):
        with open(requirements_file, "r") as file:
            for line in file:
                package = line.strip()
                if package:
                    try:
                        subprocess.check_call(["pip", "install", package])
                    except subprocess.CalledProcessError as e:
                        print(f"Failed to install {package}: {e}")
    else:
        print("requirements.txt not found.")


if __name__ == "__main__":
    install_requirements()
