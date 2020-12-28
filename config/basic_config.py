# encoding:utf-8
import os

BASE_DIR = "../data"

configs = {
    'train_data_path': os.path.join(BASE_DIR, 'train.json'),
    'val_data_path': os.path.join(BASE_DIR, 'dev.json'),
    'test_data_path': os.path.join(BASE_DIR, 'test.jon'),
    'all_data_path': os.path.join(BASE_DIR, 'all.txt')
}
