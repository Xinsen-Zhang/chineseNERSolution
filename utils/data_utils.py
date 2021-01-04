# encoding:utf-8

import os
import codecs
import json
from typing import List, Dict, Optional
import numpy as np
from torch.utils.data import RandomSampler, Dataset, DataLoader
import torch
from tqdm import tqdm
from logging import Logger
import random
from .os_utils import make_dirs
from .log_utils import init_logger
from .vocabulary_utils import Vocabulary
from logging import getLogger


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

        self.build_from_word2vec(self.vocab_path)

    def build_from_word2vec(self, text_path):
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
        if id2word[0] != '<unk>':
            raise ValueError("<unk> must map from index 0")
        if id2word[1] != '<pad>':
            raise ValueError("<pad> must map from index 1")
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


class ClunerProcessor(object):
    """
    Processor for the chinese ner data set
    """

    def __init__(self, data_dir):
        self.vocab = Vocabulary()
        self.data_dir = data_dir

    def _create_examples(self, input_path, mode):
        """
        返回 list of dict, id, context, tags, row_tags
        """
        examples = []
        with codecs.open(input_path, 'r') as f:
            idx = 0
            for line in f:
                json_data = {}
                line = json.loads(line.strip())
                text = line['text']
                label_entities = line.get("label", None)
                words = list(text)
                labels = ['O']*len(words)
                # {"organization": {"曼联": [[23, 24]]}, "name": {"温格": [[0, 1]]}}
                if label_entities is not None:
                    for key, value in label_entities.items():
                        # {"曼联": [[23, 24]]}
                        for sub_name, sub_index in value.items():
                            # [[23, 24]]
                            for start_index, end_index in sub_index:
                                assert ''.join(
                                    words[start_index:end_index+1]) == sub_name
                                if start_index == end_index:
                                    labels[start_index] = 'S-{}'.format(
                                        key)
                                else:
                                    labels[start_index] = 'B-{}'.format(
                                        key)
                                    # print(start_index, end_index, ['I-{}'.format(key)] * (len(sub_name)-1))
                                    # print(label_entities)
                                    # print(label_entities[start_index+1:end_index + 1])
                                    labels[start_index+1:end_index +
                                           1] = ['I-{}'.format(key)] * (len(sub_name)-1)
                json_data['id'] = f"{mode}-{idx}"
                json_data['context'] = " ".join(words)
                json_data['tags'] = " ".join(labels)
                json_data['raw_text'] = "".join(words)
                idx += 1
                examples.append(json_data)
        return examples

    def get_train_examples(self):
        return self._create_examples(os.path.join(self.data_dir, "train.json"), "train")

    def get_dev_examples(self):
        return self._create_examples(os.path.join(self.data_dir, "dev.json"), "dev")

    def get_test_examples(self):
        return self._create_examples(os.path.join(self.data_dir, "test.json"), "test")

    def get_vocab(self):
        vocab_path = os.path.join(self.data_dir, "vocab.pkl")
        if os.path.exists(vocab_path):
            self.vocab.load_from_file(vocab_path)
        else:
            files = ["train.json", "dev.json", "test.json"]
            for file in files:
                filename = os.path.join(self.data_dir, file)
                with codecs.open(filename, 'r', encoding='utf8') as f:
                    for line in f:
                        line = json.loads(line.strip())
                        text = line['text']
                        self.vocab.update(list(text))
            self.vocab.build_vocab()
            self.vocab.save(vocab_path)


class ClunerDataLoader(object):
    def __init__(self, data: List[Dict], batch_size: int, shuffle: bool,
                 vocab: Vocabulary, label2id: dict, seed: int, sort=True):
        self.data = data
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.seed = seed
        self.sort = sort
        self.vocab = vocab
        self.lable2id = label2id
        self._reset()

    def _reset(self):
        self.examples = self.preprocess(self.data)
        if self.sort:
            self.examples = sorted(
                self.examples, key=lambda x: x[2], reverse=True)
        if self.shuffle:
            indices = [i for i in range(len(self.examples))]
            random.shuffle(indices)
            self.examples = [self.examples[i] for i in indices]
        self.features = [self.examples[i:min(i+self.batch_size, len(self.examples))]
                         for i in range(0, len(self.examples), self.batch_size)]
        print(f"{len(self.features)} batches created")

    def preprocess(self, data_list):
        """
        Preprocess the data and convert to ids.
        """
        processed = []
        for data in data_list:  # data (dict), raw_text, context, tags, id
            text = data['context']
            tags = data['tags']
            tokens = [self.vocab.to_index(word) for word in text.split(" ")]
            x_len = len(tokens)
            tag_ids = [self.lable2id[label] for label in tags.split(" ")]
            processed.append((tokens, tag_ids, x_len, text, tags))
        return processed

    def get_long_tensor(self, token_list, batch_size, mask: Optional[bool] = True):
        """
        Convert list of list of tokens to a paded LongTensor.
        """
        token_len = max(len(x) for x in token_list)
        tokens = torch.LongTensor(batch_size, token_len).fill_(0)
        mask_ = torch.LongTensor(batch_size, token_len).fill_(0)
        for i, s in enumerate(token_list):
            tokens[i, :len(s)] = torch.LongTensor(s)
            if mask:
                mask_[i, :len(s)] = torch.tensor([1]*len(s), dtype=torch.long)
        if mask:
            return tokens, mask_
        return tokens

    def sort_all(self, batch, lens):
        """
        Sort all fields by descending order of lens, and return the original indices.
        奇淫巧技
        """
        unsorted_all = [lens] + [[i for i in range(len(lens))]] + list(batch)
        sorted_all = [list(t) for t in zip(
            *sorted(zip(*unsorted_all), reverse=True))]
        return sorted_all[2:], sorted_all[1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        """
        Get a batch with an index.
        """
        if not isinstance(index, int):
            raise TypeError(
                "expect type of index is int, but get {}".format(type(index).__name__))
        if index < 0 or index >= len(self.features):
            raise IndexError("index out of bounds")
        batch = self.features[index]
        batch_size = len(batch)
        batch = list(zip(*batch))
        # batch[0] is list of tokens
        # batch[1] is list of tag_ids
        # batch[2] is list of length of tokens
        # batch[3] is list of string containing text
        # batch[4] is list of string containing tags
        lens = [len(x) for x in batch[0]]
        batch, origin_index = self.sort_all(batch, lens)
        token_list = batch[0]
        input_ids, input_mask = self.get_long_tensor(
            token_list, batch_size, mask=True)
        label_ids = self.get_long_tensor(batch[1], batch_size, mask=False)
        input_lens = [len(x) for x in batch[0]]
        return (input_ids, input_mask, label_ids, input_lens)


class NerDataSet(Dataset):
    def __init__(self, processor: ClunerProcessor, mode: str = 'train', max_length: int = 50,
                 label2id: dict = None):
        self.processor = processor
        self.mode = mode
        self.max_length = max_length
        self.label2id = label2id
        self.processor.get_vocab()
        if self.mode == 'train':
            self.examples = self.processor.get_train_examples()
        elif self.mode == 'test':
            self.examples = self.processor.get_test_examples()
        else:
            self.examples = self.processor.get_dev_examples()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        """

        Args:
            index ([int]): 数据的 index

        Returns:
            input_ids, input_mask, label_ids, input_lens
        """
        item = self.examples[index]
        tokens = item['context']  # "今 天 天 气 真 好 啊"
        tags = item['tags']  # "B-PER I-PER S-PER O O O"
        tokens = tokens.split(" ")
        tags = tags.split(" ")
        token_ids = [self.processor.vocab.to_index(token) for token in tokens]
        tag_ids = [self.label2id[tag] for tag in tags]
        token_lens = len(token_ids)
        mask = [1 for i in range(token_lens)]
        if len(token_ids) < self.max_length:
            for _ in range(self.max_length-len(token_ids)):
                mask.append(0)
                token_ids.append(self.processor.vocab.padding_idx())
                tag_ids.append(self.label2id['O'])
        return (torch.tensor(token_ids, dtype=torch.long),
                torch.tensor(mask, dtype=torch.long),
                torch.tensor(tag_ids, dtype=torch.long),
                torch.tensor(token_lens, dtype=torch.long))
                # token_lens)


class NerDataLoader:
    def __init__(self, mode: str = 'train', batch_size: int = 128, drop_last: bool = False, max_length: int = 50,
                 shuffle: bool = False, logger: Logger = None, data_dir: str = None, num_workers: int = 8, label2id: dict = None):
        if data_dir is None or len(data_dir) == 0:
            raise ValueError(
                "data dir expect of a valid string, but got None or empty string")
        if logger is None:
            self.logger = getLogger()
        else:
            self.logger = logger
        self.processor = ClunerProcessor(data_dir)
        self.mode = mode
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.max_length = max_length
        self.label2id = label2id
        self.dataset = self._get_dataset()

    def _get_dataset(self):
        return NerDataSet(self.processor, self.mode, max_length=self.max_length,
                          label2id=self.label2id)

    def get_dataloader(self):
        if self.shuffle:
            sampler = RandomSampler(self.dataset)
        else:
            sampler = None
        if sampler is None:
            return DataLoader(self.dataset, batch_size=self.batch_size,
                              num_workers=self.num_workers, drop_last=self.drop_last)
        else:
            return DataLoader(self.dataset, batch_size=self.batch_size,
                              num_workers=self.num_workers, drop_last=self.drop_last, sampler=sampler)


if __name__ == "__main__":
    with codecs.open("../data/train.json") as f:
        for line in f:
            item = json.loads(line)
            print(item)
