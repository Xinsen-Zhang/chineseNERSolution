from utils.log_utils import init_logger
from utils.data_utils import DataTransformer, NERDataLoader
from config.basic_config import configs
import codecs

import os

if __name__ == "__main__":
    logger = init_logger(
        log_name='main', log_dir=configs['log_dir'])
    # 将数据保存起来
    # save_data(configs['all_data_path'],
    #           configs['train_data_path'], configs['test_data_path'],
    #           configs['val_data_path'])
    # 构建 transformers
    transformer = DataTransformer(os.path.join(
        configs['embedding_dir'], 'char_word2vec_size200_win5.txt'), logger,
        all_data_path=configs['all_data_path'], tag2id=configs['tag2id'],
        max_length=configs['max_length'])
    # 获取 word2id 和 embedding_matrix
    logger.info("loading word2id and embedding matrix")
    word2id = transformer.word2id
    embedding_matrix = transformer.embedding_matrix
    # for word, index in word2id.items():
    #     logger.info(f"word: {word}, vector: {embedding_matrix[index]}")
    with codecs.open(configs['ptrain_x_path']) as f:
        x = f.readlines()
    with codecs.open(configs['ptrain_y_path']) as f:
        y = f.readlines()
    line_count = 0
    for x_item, y_item in zip(x, y):
        logger.info('train data, line: {}'.format(line_count+1))
        x_item = x_item.rstrip().split(" ")
        y_item = y_item.rstrip().split(" ")
        encoded_sentence, encoded_tags, mask = transformer.\
            encode_pairs(x_item,
                         y_item,
                         return_mask=True)
        for char_index, tag_index, mask_value in zip(encoded_sentence,
                                                     encoded_tags, mask):
            logger.info(f'char: {transformer.id2word[char_index]}, index: ' +
                        f'{char_index}, tag: ' +
                        f'{configs["id2tag"][tag_index]},' +
                        f'tindex: {tag_index}, ' +
                        f'mask: {mask_value}')
        line_count += 1
        if line_count == 5:
            break
    ner_base_loader = NERDataLoader(transformer, configs['device'],
                                    configs['max_length'], logger=logger)
    logger.info("training data loader")
    loader1 = ner_base_loader.load_data(
        configs['ptrain_x_path'], configs['ptrain_y_path'], True,
        configs['batch_size'], configs['num_workers'], shuffle=True)
    for x_train, y_train, mask in loader1:
        print(x_train.shape)
        print(y_train.shape)
        x_train = x_train.detach().cpu().numpy()
        y_train = y_train.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()
        for i in range(x_train.shape[0]):
            logger.info(f"batch: {i+1}")
            for j in range(configs['max_length']):
                logger.info(f'batch:{i+1}, ' +
                            f'char:{transformer.id2word[x_train[i,j]]},' +
                            f' tag:{configs["id2tag"][y_train[i,j]]}' +
                            f' mask: {mask[i,j]}')
        break
    logger.info("validation data loader")
    loader1 = ner_base_loader.load_data(
        configs['pval_x_path'], configs['pval_y_path'], True,
        configs['batch_size'], configs['num_workers'], shuffle=True)
    # print(len(loader1))
    for x_val, y_val, mask in loader1:
        print(x_val.shape)
        print(y_val.shape)
        x_val = x_val.detach().cpu().numpy()
        y_val = y_val.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()
        for i in range(x_val.shape[0]):
            logger.info(f"batch: {i+1}")
            for j in range(configs['max_length']):
                logger.info(f'batch:{i+1}, ' +
                            f'char:{transformer.id2word[x_val[i,j]]},' +
                            f' tag:{configs["id2tag"][y_val[i,j]]}' +
                            f' mask: {mask[i,j]}')
        break

    logger.info("test data loader")
    loader1 = ner_base_loader.load_data(
        configs['ptest_x_path'], None, True,
        configs['batch_size'], configs['num_workers'], shuffle=True)
    # print(len(loader1))
    for x_test, mask in loader1:
        print(x_test.shape)
        # print(y_val.shape)
        x_test = x_test.detach().cpu().numpy()
        # y_val = y_val.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()
        for i in range(x_test.shape[0]):
            logger.info(f"batch: {i+1}")
            for j in range(configs['max_length']):
                logger.info(f'batch:{i+1}, ' +
                            f'char:{transformer.id2word[x_test[i,j]]},' +
                            # f' tag:{configs["id2tag"][y_val[i,j]]}' +
                            f' mask: {mask[i,j]}')
        break
