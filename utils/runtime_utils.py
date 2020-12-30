# encoding:utf-8

import os
import torch
import numpy as np
import random
from .log_utils import init_logger
from logging import Logger
from typing import Optional
from pathlib import Path
from torch import nn


def print_config(config, logger: Optional[Logger] = None):
    if logger is None:
        logger = init_logger("runtime_utils", "./")
    info = "Running with the following configs:\n"
    for k, v in config.items():
        info += f"\t{k} : {str(v)}\n"
    print("\n" + info + "\n")
    return


def seed_everything(seed: Optional[int] = 1029) -> None:
    """设置整个开发环境的seed

    Args:
        seed (Optional[int], optional): 随机种子. Defaults to 1029.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def prepare_device(n_gpu_use=None, logger: Optional[Logger] = None):
    """
        setup GPU device if available

    Args:
        n_gpu_use (int, List[int], None):   None for cpu use
                                            int for a list via range
                                            else make list[0] as controller
        logger (Optional[Logger], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """

    if logger is None:
        logger = init_logger("runtime_utils", "./")

    if not n_gpu_use:
        device_type = 'cpu'
    else:
        n_gpu_use = [i for i in range(n_gpu_use)] if type(
            n_gpu_use) == int else n_gpu_use
        device_type = f"cuda:{n_gpu_use[0]}"
    n_gpu = torch.cuda.device_count()
    if len(n_gpu_use) > 0 and n_gpu == 0:
        logger.warning(
            "Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        device_type = 'cpu'
    if len(n_gpu_use) > n_gpu:
        msg = f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are available on this machine."
        logger.warning(msg)
        n_gpu_use = range(n_gpu)
    device = torch.device(device_type)
    list_ids = n_gpu_use
    return device, list_ids


def model_device(n_gpu, model, logger: Optional[Logger] = None):
    '''
    判断环境 cpu还是gpu
    支持单机多卡
    :param n_gpu:
    :param model:
    :return:
    '''
    if logger is None:
        logger = init_logger("runtime_utils", "./")
    device, device_ids = prepare_device(n_gpu)
    if len(device_ids) > 1:
        logger.info(f"current {len(device_ids)} GPUs")
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    if len(device_ids) == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_ids[0])
    model = model.to(device)
    return model, device


def restore_checkpoint(resume_path, model=None):
    '''
    加载模型
    :param resume_path:
    :param model:
    :param optimizer:
    :return:
    注意： 如果是加载Bert模型的话，需要调整，不能使用该模式
    可以使用模块自带的Bert_model.from_pretrained(state_dict = your save state_dict)
    '''
    if isinstance(resume_path, Path):
        resume_path = str(resume_path)
    checkpoint = torch.load(resume_path)
    best = checkpoint['best']  # 取出best
    start_epoch = checkpoint['epoch'] + 1
    states = checkpoint['state_dict']
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(states)
    else:
        model.load_state_dict(states)
    return [model, best, start_epoch]
