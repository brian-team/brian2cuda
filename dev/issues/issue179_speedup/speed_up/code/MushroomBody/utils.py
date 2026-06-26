import __main__
import os
import shutil


def get_directory(device, basedir='code', delete_dir=False):
    name = os.path.splitext(os.path.basename(__main__.__file__))[0]
    directory = os.path.join(basedir, name, device)
    if delete_dir:
        shutil.rmtree(directory, ignore_errors=True)
    return directory


if __name__ == "__main__":
    print(get_directory("cuda_standalone"))