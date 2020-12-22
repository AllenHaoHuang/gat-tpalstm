import os


def check_path_exists(path):
    return os.path.exists(path)


def create_dir(path):
    try:
        os.makedirs(path)
    except os.error:
        pass
