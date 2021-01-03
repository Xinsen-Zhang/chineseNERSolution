# encoding:utf-8

import os
from pathlib import Path
import pickle
from torch import nn
import torch


def make_dirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_pickle(data, file_path):
    """保存成pickle文件

    Args:
        data ([obejct]): 待保存的内容
        file_path ([str]): 序列化保存至文件的文件名
    """
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


def save_model(model: nn.Module, model_path):
    if isinstance(model, nn.DataParallel):
        model = model.modules
    state_dict = model.state_dict()
    for key, value in state_dict.items():
        state_dict[key] = value.cpu()
    torch.save(state_dict, model_path)


def load_model(model: nn.Module, model_path):
    state_dict = torch.load(model_path)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    if isinstance(model, nn.DataParallel):
        model.modules.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    return model
