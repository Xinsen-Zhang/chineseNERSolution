# encoding:utf-8
import os

BASE_DIR = "./data"
labels = [
    "address",
    "book",
    "company",
    "game",
    "goverment",
    "movie",
    "name",
    "organization",
    "position",
    "scene"
]


def generate_tag2id(labels):
    result = {'O': 0}
    for index, label in enumerate(labels):
        result[f"B_{label}"] = 2*index+1
        result[f"I_{label}"] = 2*index+2
    return result


tag2id = generate_tag2id(labels)
id2tag = {v: k for k, v in tag2id.items()}

configs = {
    # data config
    'train_data_path': os.path.join(BASE_DIR, 'train.json'),
    'val_data_path': os.path.join(BASE_DIR, 'dev.json'),
    'test_data_path': os.path.join(BASE_DIR, 'test.json'),
    'all_data_path': os.path.join(BASE_DIR, 'all.txt'),
    "ptrain_x_path": os.path.join(BASE_DIR, "x_train.txt"),
    "pval_x_path": os.path.join(BASE_DIR, "x_val.txt"),
    "ptest_x_path": os.path.join(BASE_DIR, "x_test.txt"),
    'ptrain_y_path': os.path.join(BASE_DIR, 'y_train.txt'),
    'pval_y_path': os.path.join(BASE_DIR, 'y_val.txt'),
    # 'p_train_y_path': os.path.join(BASE_DIR, 'y_train.txt'),



    # word2vec config
    'embedding_dir': os.path.join(BASE_DIR, "../embeddings/"),

    # log config
    "log_dir": os.path.join(BASE_DIR, "../logs/"),

    # data-loader parameters
    "batch_size": 128,
    "max_length": 49,
    "num_workers": 8,

    # label config
    "labels": labels,
    "tag2id": tag2id,
    'id2tag': id2tag
}
