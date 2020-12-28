# encoding:utf-8

import codecs
import json
from typing import List, Dict
from .os_utils import make_dirs


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
    with codecs.open("../data/train.json") as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data


def save_data(data_filepath: str, train_filepath: str, test_filepath: str)\
        -> None:
    """
    将所有的数据保存下来
    """
    train_data = get_data(train_filepath)
    test_data = get_data(test_filepath)
    all_data = []
    for item in train_data:
        all_data.append(item['text'])
    for item in test_data:
        all_data.append(item['text'])
    data_dir = "/".join(data_filepath.split('/')[:-1])
    make_dirs(data_dir)
    with codecs.open(data_filepath, encoding='utf8', mode='w') as f:
        for line in all_data:
            f.write(f"{line}\n")


if __name__ == "__main__":
    with codecs.open("../data/train.json") as f:
        for line in f:
            item = json.loads(line)
            print(item)
