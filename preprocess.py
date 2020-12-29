# encoding:utf-8
from utils.log_utils import init_logger
from utils.data_utils import save_data, json2text, get_data, write_text
from config.basic_config import configs
import codecs
import numpy as np

if __name__ == "__main__":
    logger = init_logger(
        log_name='preprocess', log_dir=configs['log_dir'])
    # 将数据保存起来
    save_data(configs['all_data_path'],
              configs['train_data_path'], configs['test_data_path'],
              configs['val_data_path'])
    with codecs.open(configs['all_data_path']) as f:
        content = f.readlines()
    length_list = []
    for line in content:
        line = line.strip()
        if len(line) == 0:
            continue
        length_list.append(len(line))
    # print(length_list)
    max_length = int(np.percentile(length_list, 95))
    logger.info(f"max length: {max_length}")
    json2text(configs['test_data_path'], configs['ptest_x_path'], key='text')
    train_data = get_data(configs['train_data_path'])
    val_data = get_data(configs['val_data_path'])
    train_collections = []
    val_collections = []
    for item in train_data:
        tags = ['O' for _ in range(len(item['text']))]
        label = item['label']
        for tag, info in label.items():
            temp = list(info.values())[0][0]
            from_index = temp[0]
            to_index = temp[1]
            for index in range(from_index, min(to_index+1, len(tags))):
                tags[index] = f"I_{tag}"
            tags[from_index] = f"B_{tag}"
        # tags = "".join(tags)
        train_collections.append((item['text'], tags))

    for item in val_data:
        tags = ['O' for _ in range(len(item['text']))]
        label = item['label']
        for tag, info in label.items():
            temp = list(info.values())[0][0]
            from_index = temp[0]
            to_index = temp[1]
            for index in range(from_index, min(to_index+1, len(tags))):
                tags[index] = f"I_{tag}"
            tags[from_index] = f"B_{tag}"
        # tags = "".join(tags)
        val_collections.append((item['text'], tags))
    write_text(configs['ptrain_x_path'], [item[0]
                                          for item in train_collections])
    write_text(configs['ptrain_y_path'], [item[1]
                                          for item in train_collections])
    write_text(configs['pval_x_path'], [item[0] for item in val_collections])
    write_text(configs['pval_y_path'], [item[1] for item in val_collections])
    logger.info("have a look at a sample train data")
    with codecs.open(configs['ptrain_x_path']) as f:
        x = f.readlines()
    with codecs.open(configs['ptrain_y_path']) as f:
        y = f.readlines()
    for x_item, y_item in zip(x, y):
        x_item = x_item.rstrip().split(" ")
        y_item = y_item.rstrip().split(" ")
        for char, tag in zip(x_item, y_item):
            logger.info(f"char: {char}, tag: {tag}")
        break
    logger.info("have a look at a sample dev data")
    with codecs.open(configs['pval_x_path']) as f:
        x = f.readlines()
    with codecs.open(configs['pval_y_path']) as f:
        y = f.readlines()
    for x_item, y_item in zip(x, y):
        x_item = x_item.rstrip().split(" ")
        y_item = y_item.rstrip().split(" ")
        for char, tag in zip(x_item, y_item):
            logger.info(f"char: {char}, tag: {tag}")
        break
