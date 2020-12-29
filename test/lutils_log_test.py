# encoding:utf-8

import sys
import os

sys.path.append(os.path.join(os.getcwd(), "../"))
from utils.log_utils import init_logger



if __name__ == "__main__":
    logger = init_logger('test', './data')
    logger.info("test")
    logger.error("test2")
