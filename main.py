from utils.log_utils import init_logger
from utils.data_utils import DataTransformer, save_data
from config.basic_config import configs

import os

if __name__ == "__main__":
    logger = init_logger(
        log_name='utils', log_dir=configs['log_dir'])
    # 将数据保存起来
    # save_data(configs['all_data_path'],
    #           configs['train_data_path'], configs['test_data_path'],
    #           configs['val_data_path'])
    # 构建 transformers
    transformer = DataTransformer(os.path.join(
        configs['embedding_dir'], 'char_word2vec_size200_win5.txt'), logger,
        all_data_path=configs['all_data_path'])
    # 获取 word2id 和 embedding_matrix
    logger.info("loading word2id and embedding matrix")
    word2id = transformer.word2id
    embedding_matrix = transformer.embedding_matrix
