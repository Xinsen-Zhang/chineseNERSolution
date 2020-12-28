# encoding:utf-8

import sys
import os

sys.path.append(os.path.join(os.getcwd(), "../"))
from utils.data_utils import get_data, save_data
from config.basic_config import configs


if __name__ == "__main__":
    save_data(configs['all_data_path'],
              configs['train_data_path'], configs['test_data_path'])
