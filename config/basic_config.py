# encoding:utf-8
import os
import torch

BASE_DIR = os.path.join("/".join(os.path.abspath(__file__).split("/")[:-1]), "../data")
labels = [
    "address",
    "book",
    "company",
    "game",
    "government",
    "movie",
    "name",
    "organization",
    "position",
    "scene"
]


def generate_tag2id(labels):
    result = {'O': 0}
    for index, label in enumerate(labels):
        result[f"B-{label}"] = 3*index+1
        result[f"I-{label}"] = 3*index+2
        result[f"S-{label}"] = 3*index+3
    return result


tag2id = generate_tag2id(labels)
tag2id['<START>'] = len(tag2id)
tag2id['<STOP>'] = len(tag2id)
id2tag = {v: k for k, v in tag2id.items()}

configs = {
    # data config
    "base_dir": BASE_DIR,
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
    'id2tag': id2tag,

    # pytorch config
    "device": torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device('cpu'),

    # model config
    "batch_first": True,
    # embedding config
    "dropout_emb": 0.1,
    "is_frozen": True,
    # lstm config
    "hidden_size": 512,
    "num_layers": 2,
    "dropout_lstm": 0.1,
    "is_bidirectional": True,

    # optimizer config
    "learning_rate": 0.01,
    "weight_decay": 5e-4,

    # task config
    "num_classes": 21,
    "crf_num_classes": 23,

    # output_dir
    "output_dir": os.path.join(BASE_DIR, '../checkpoints/'),
}

# print(__file__)
# print(BASE_DIR)