import subprocess
import sys

from src.services.facescan.plugins.managers import plugin_manager


def install_requirements(requirements: set):
    print(f'Install dependencies: {requirements}')
    cmd = f"{sys.executable} -m pip install --no-cache-dir {' '.join(requirements)}"
    try:
        subprocess.run(cmd.split(), check=True)
    except subprocess.CalledProcessError:
        exit(1)


if __name__ == '__main__':
    install_requirements(plugin_manager.requirements)

    for plugin in plugin_manager.plugins:
        if plugin.ml_model:
            print(f'Checking models for {plugin}...')
            try:
                plugin.ml_model.download_if_not_exists()
            except Exception as e:
                print(f'Warning: Could not download model for {plugin}: {e}')
                print('This is expected during build and will be resolved when the container runs')
