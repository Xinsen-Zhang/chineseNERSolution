# encoding:utf-8

import codecs
import json
from typing import List, Dict, Optional
from .os_utils import make_dirs
import numpy as np
from torch.utils.data import RandomSampler, Dataset, DataLoader
import torch


def get_data(data_filepath: str) -> List[Dict]:
    """根据路径获得数据

    Args:
        data_filepath (str): 数据所在路径

    Returns:
        List[Dict]: 数据
                    dict: {'text': '至尊宝——10名《夺宝冒险王》游戏',
                    'label': {'game': {'《夺宝冒险王》': [[8, 14]]}}}
    """
    data = []
    with codecs.open(data_filepath) as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data


def save_data(data_filepath: str, train_filepath: str, test_filepath: str,
              val_filepath: str = None)\
        -> None:
    """
    将所有的数据保存下来
    """
    # print(train_filepath,val_filepath)
    train_data = get_data(train_filepath)
    test_data = get_data(test_filepath)
    if val_filepath is not None:
        val_data = get_data(val_filepath)
    all_data = []
    for item in train_data:
        all_data.append(item['text'])
    for item in test_data:
        all_data.append(item['text'])
    if val_filepath is not None:
        for item in val_data:
            all_data.append(item['text'])
    data_dir = "/".join(data_filepath.split('/')[:-1])
    make_dirs(data_dir)
    with codecs.open(data_filepath, encoding='utf8', mode='w') as f:
        for line in all_data:
            f.write(f"{line}\n")


def write_text(filename: str, content: List[str], is_append:
               Optional[bool] = False) -> None:
    if is_append:
        with codecs.open(filename, mode='a', encoding='utf8') as f:
            for line in content:
                if type(line) == str:
                    f.write("{}\n".format(" ".join([char for char in line])))
                elif type(line) == list:
                    f.write("{}\n".format(" ".join(line)))
    else:
        with codecs.open(filename, mode='w', encoding='utf8') as f:
            for line in content:
                if type(line) == str:
                    f.write("{}\n".format(" ".join([char for char in line])))
                elif type(line) == list:
                    f.write("{}\n".format(" ".join(line)))


def json2text(datapath: str, dest_filepath: str, is_append:
              Optional[bool] = False, key: Optional[str] = None) -> None:
    data = get_data(datapath)
    if key is not None:
        content = [item[key] for item in data]
    else:
        content = [json.dumps[item] for item in data]
    write_text(dest_filepath, content, is_append)


class DataTransformer(object):
    def __init__(self,
                 vocab_path,
                 logger,
                 max_features=None,
                 min_freq=3,
                 all_data_path=None,
                 label_to_id=None,
                 train_file=None,
                 valid_file=None,
                 test_file=None,
                 valid_size=None,
                 skip_header=False,
                 is_train_mode=True,
                 seed=1024):
        self.seed = seed
        self.logger = logger
        self.valid_size = valid_size
        self.min_freq = min_freq
        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = test_file
        self.all_data_path = all_data_path
        self.vocab_path = vocab_path
        self.skip_header = skip_header
        self.max_features = max_features
        self.label_to_id = label_to_id
        self.is_train_mode = is_train_mode

        self.build_vocabs(self.all_data_path)
        self.embedding_matrix = self.build_embedding_matrix(self.vocab_path)

    def build_vocabs(self, text_path):
        #  = self.all_data_path
        all_words = []
        word2count = {}
        with codecs.open(text_path) as f:
            for line in f:
                line = line.strip().split(" ")
                for word in line:
                    c = word2count.get(word, 0)
                    c += 1
                    word2count[word] = c
        word2count = sorted(word2count.items(),
                            key=lambda x: x[1], reverse=True)
        all_words = [_[0] for _ in word2count]
        all_words = ['<pad>', '<unk>'] + all_words
        self.id2word = dict(enumerate(all_words))
        self.word2id = {v: k for k, v in enumerate(self.id2word)}

    def _load_embedding(self, embedding_path):
        '''
        加载pretrained
        :param embedding_path:
        :return:
        '''
        self.logger.info("load emebedding weights")
        embeddings_index = {}
        f = open(embedding_path, 'r', errors='ignore', encoding='utf8')
        for line in f:
            values = line.split(' ')
            try:
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            except Exception:
                print("Error on ", values[:2])
        f.close()
        self.logger.info('Total %s word vectors.' % len(embeddings_index))
        return embeddings_index

    def build_embedding_matrix(self, embedding_path, emb_mean=None,
                               emb_std=None):
        '''
        构建词向量权重矩阵
        :param embedding_path:
        :return:
        '''
        self.logger.info("initializer embedding matrix")
        embeddings_index = self._load_embedding(embedding_path)
        all_embs = np.stack(list(embeddings_index.values()))
        if emb_mean is None or emb_std is None:
            emb_mean = all_embs.mean()
            emb_std = all_embs.std()
        embed_size = all_embs.shape[1]
        nb_words = len(self.word2id)
        # 这里我们简单使用正态分布产生随机值
        embedding_matrix = np.random.normal(
            emb_mean, emb_std, (nb_words, embed_size))
        for word, id in self.word2id.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[id] = embedding_vector

        return embedding_matrix


class NERDataSet(Dataset):
    def __init__(self, x: torch.LongTensor, y: torch.LongTensor,
                 device: torch.device, max_length: int = 128):
        self.device = device
        self.x = x
        self.y = y
        self.max_length = max_length

    def __getitem__(self, index):
        sentence_tensor = self.x[index]
        tag_tensor = self.y[index]
        mask = torch.ones((1, self.max_length))
        length = sentence_tensor.size()[0]
        if length >= self.max_length:
            start_index = np.random.randint(0, length-self.max_length)
            return sentence_tensor[start_index:
                                   start_index +
                                   self.max_length].to(self.deivce),
            tag_tensor[start_index:start_index +
                       self.max_length].to(self.device),
            mask.to(self.device)

    def __len__(self):
        return len(self.x)


if __name__ == "__main__":
    with codecs.open("../data/train.json") as f:
        for line in f:
            item = json.loads(line)
            print(item)
