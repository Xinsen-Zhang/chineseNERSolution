# encoding:utf-8

import os
from pathlib import Path
import pickle


def make_dirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_pickle(data, file_path):
    '''
    保存成pickle文件
    :param data:
    :param file_name:
    :param pickle_path:
    :return:
    '''
    if isinstance(file_path, Path):
        file_path = str(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(input_file) -> object:
    """
        读取pickcle文件
    Args:
        input_file (str or Path): 读取文件的路径

    Returns:
        [object]: pickle内容
    """
    with open(str(input_file), 'rb') as f:
        data = pickle.load(f)
    return data
