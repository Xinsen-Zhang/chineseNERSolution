# encoding:utf-8

import codecs
import json
from typing import List, Dict, Optional
from .os_utils import make_dirs
import numpy as np
from torch.utils.data import RandomSampler, Dataset, DataLoader
import torch
from tqdm import tqdm
from logging import Logger
from .log_utils import init_logger


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
                 max_length: Optional[int] = None,
                 min_freq=3,
                 all_data_path=None,
                 #  label_to_id=None,
                 train_file=None,
                 valid_file=None,
                 test_file=None,
                 valid_size=None,
                 skip_header=False,
                 is_train_mode=True,
                 tag2id: Optional[Dict[str, int]] = None,
                 seed: Optional[int] = 1024):
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
        self.max_length = max_length
        # self.label_to_id = label_to_id
        self.is_train_mode = is_train_mode
        self.tag2id = tag2id

        # self.build_vocabs(self.all_data_path)
        self.build_from_word2vec(self.vocab_path)

    def build_from_word2vec(self, text_path):
        #  = self.all_data_path
        # all_words = []
        # word2count = {}
        # with codecs.open(text_path) as f:
        #     for line in f:
        #         line = line.strip().split(" ")
        #         for word in line:
        #             c = word2count.get(word, 0)
        #             c += 1
        #             word2count[word] = c
        # word2count = sorted(word2count.items(),
        #                     key=lambda x: x[1], reverse=True)
        # all_words = [_[0] for _ in word2count]
        # all_words = ['<pad>', '<unk>'] + all_words
        # self.id2word = dict(enumerate(all_words))
        # self.word2id = {v: k for k, v in enumerate(self.id2word)}
        self.word2id, self.embedding_matrix = \
            self.build_embedding_matrix(self.vocab_path)
        self.id2word = {v: k for k, v in self.word2id.items()}

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
        nb_words = all_embs.shape[0]+2
        # 这里我们简单使用正态分布产生随机值
        all_words = ['<unk>', '<pad>']
        for word, _ in embeddings_index.items():
            all_words.append(word)
        id2word = dict(enumerate(all_words))
        word2id = {v: k for k, v in id2word.items()}
        embedding_matrix = np.random.normal(
            emb_mean, emb_std, (nb_words, embed_size))
        for word, embedding_vector in embeddings_index.items():
            # embedding_vector = embeddings_index.get(word)
            # if embedding_vector is not None:
            embedding_matrix[word2id[word]] = embedding_vector

        return word2id, embedding_matrix

    def encode_sentence(self, sentence: List[str],
                        return_mask: Optional[bool] = False,
                        randomize: Optional[bool] = False):
        length = len(sentence)
        if randomize:
            # TODO 随机采样
            pass
        encoded_sentence = [self.word2id[word] if word in self.word2id else
                            self.word2id['<unk>'] for word in sentence]
        if length > self.max_length:
            encoded_sentence = encoded_sentence[:self.max_length]
        else:
            for _ in range(self.max_length-length):
                encoded_sentence.append(self.word2id['<pad>'])
        if return_mask:
            mask = [0 for i in range(self.max_length)]
            if length <= self.max_length:
                for i in range(length):
                    mask[i] = 1
            else:
                for i in range(self.max_length):
                    mask[i] = 1
            return encoded_sentence, mask
        return encoded_sentence

    def encode_tags(self, tags: List[str]) -> List[str]:
        return [self.tag2id[tag] for tag in tags]

    def encode_pairs(self, sentence: List[str], tags: List[str],
                     return_mask: Optional[bool] = False,
                     randomize: Optional[bool] = False):
        length = len(sentence)
        if randomize:
            # TODO 随机采样
            pass
        encoded_sentence = [self.word2id[word] if word in self.word2id else
                            self.word2id['<unk>'] for word in sentence]
        encoded_tags = [self.tag2id[tag] for tag in tags]
        if length > self.max_length:
            encoded_sentence = encoded_sentence[:self.max_length]
            encoded_tags = encoded_tags[:self.max_length]
        else:
            for _ in range(self.max_length-length):
                encoded_sentence.append(self.word2id['<pad>'])
                encoded_tags.append(self.tag2id['O'])
        if return_mask:
            mask = [0 for i in range(self.max_length)]
            if length <= self.max_length:
                for i in range(length):
                    mask[i] = 1
            else:
                for i in range(self.max_length):
                    mask[i] = 1
            return encoded_sentence, encoded_tags, mask
        return encoded_sentence, encoded_tags


class NERDataSet(Dataset):
    def __init__(self, x: List[List[str]], y: List[List[str]],
                 #  device: torch.device,
                 transformer: DataTransformer,
                 max_length: int = 128,
                 return_mask: Optional[bool] = False,
                 is_training: Optional[bool] = False):
        self.device = torch.device("cpu")
        self.x = x
        self.y = y
        self.max_length = max_length
        self.return_mask = return_mask
        self.is_training = is_training
        self.transformer = transformer

    def __getitem__(self, index):
        sentence = self.x[index]
        if not self.is_training:
            result = self.transformer.encode_sentence(
                sentence, self.return_mask)
            if self.return_mask:
                encoded_sentence = torch.LongTensor(result[0],
                                                    # dtype=torch.LongTensor,
                                                    device=self.device)
                masks = torch.LongTensor(result[1],
                                         #  dtype=torch.LongTensor,
                                         device=self.device)
                return encoded_sentence, masks
            else:
                encoded_sentence = torch.LongTensor(result,
                                                    # dtype=torch.LongTensor,
                                                    device=self.device)
                return encoded_sentence
        else:
            tags = self.y[index]
            result = self.transformer.encode_pairs(
                sentence, tags, self.return_mask)
            if self.return_mask:
                encoded_sentence = torch.LongTensor(result[0],
                                                    # dtype=torch.LongTensor,
                                                    device=self.device)
                encoded_tags = torch.LongTensor(result[1],
                                                # dtype=torch.LongTensor,
                                                device=self.device)
                masks = torch.LongTensor(result[2],
                                         #  dtype=torch.LongTensor,
                                         device=self.device)
                return encoded_sentence, encoded_tags, masks
            else:
                encoded_sentence = torch.LongTensor(result[0],
                                                    # dtype=torch.LongTensor,
                                                    device=self.device)
                encoded_tags = torch.LongTensor(result[1],
                                                # dtype=torch.LongTensor,
                                                device=self.device)
                return encoded_sentence, encoded_tags

    def __len__(self):
        return len(self.x)


class NERDataLoader(object):
    def __init__(self, transformer: DataTransformer,
                 device: torch.device,
                 max_length: Optional[int] = 128,
                 logger: Optional[Logger] = None):
        self.transformer = transformer
        self.device = device
        self.max_length = max_length
        self.logger = logger if logger is not None else\
            init_logger("nerDataLoader", "./logs")

    def load_data(self, sentence_path: str,
                  tag_path: Optional[str] = None,
                  return_mask: Optional[bool] = False,
                  batch_size: Optional[int] = 128,
                  num_workers: Optional[int] = 8,
                  shuffle: Optional[bool] = False):
        x_data = []
        if tag_path is not None:
            y_data = []
        # print(self)
        self.logger.info("loading proccessed sentence data from disk")
        with codecs.open(sentence_path) as f:
            x = f.readlines()
        self.logger.info("training mode: " +
                         f"{False if tag_path is None else True}")
        if tag_path is not None:
            self.logger.info("loading proccessed sentence data from disk")
            with codecs.open(tag_path) as f:
                y = f.readlines()
            t = tqdm(zip(x, y), total=len(x))
            for x_item, y_item in t:
                x_item = x_item.rstrip().split(" ")
                y_item = y_item.rstrip().split(" ")
                x_data.append(x_item)
                y_data.append(y_item)
            t.close()
            dataset = NERDataSet(x_data, y_data,
                                 self.transformer, self.max_length,
                                 return_mask, True)
        else:
            t = tqdm(x, total=len(x))
            for x_item in t:
                x_item = x_item.rstrip().split(" ")
                x_data.append(x_item)
            t.close()
            dataset = NERDataSet(x_data, None,
                                 self.transformer, self.max_length,
                                 return_mask, False)
        if tag_path is not None:
            sampler = RandomSampler(dataset)
            return DataLoader(dataset, sampler=sampler, batch_size=batch_size,
                              num_workers=num_workers, shuffle=False)
        else:
            return DataLoader(dataset, batch_size=batch_size,
                              num_workers=num_workers, shuffle=False)


if __name__ == "__main__":
    with codecs.open("../data/train.json") as f:
        for line in f:
            item = json.loads(line)
            print(item)
