import os.path as path


def settings_file():
    """
    :return: project settings from the 'SETTINGS.json' file
    """
    return path.join(path.dirname(path.realpath(__file__)), '.', 'SETTINGS.json')