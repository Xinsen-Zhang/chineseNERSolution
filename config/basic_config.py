# encoding:utf-8
import os

BASE_DIR = "./data"

configs = {
    'train_data_path': os.path.join(BASE_DIR, 'train.json'),
    'val_data_path': os.path.join(BASE_DIR, 'dev.json'),
    'test_data_path': os.path.join(BASE_DIR, 'test.json'),
    'all_data_path': os.path.join(BASE_DIR, 'all.txt'),

    # word2vec config
    'embedding_dir': os.path.join(BASE_DIR, "../embeddings/"),

    # log config
    "log_dir": os.path.join(BASE_DIR, "../logs/")
}
