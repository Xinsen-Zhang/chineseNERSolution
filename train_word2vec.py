# encoding:utf-8
import argparse
import codecs
from config.basic_config import configs
# from utils import log_utils
from utils.log_utils import init_logger
from models.embedding import word2vec


def main(**kwargs):
    logger = init_logger(
        log_name='word2vec', log_dir=configs['log_dir'])
    logger.info('load %s data from disk ' % kwargs['tag'])
    data = []
    with codecs.open(configs['all_data_path']) as f:
        for line in f:
            data.append(line)

    logger.info("initializing emnedding model")
    word2vec_model = word2vec.Word2Vec(size=kwargs['size'],
                                       window=kwargs['window'],
                                       min_count=3,
                                       tag=kwargs['tag'],
                                       save_dir=configs['embedding_dir'],
                                       logger=logger)
    logger.info('train %s word2vec embedding' % kwargs['tag'])
    tag = kwargs['tag']
    if tag == 'word':
        word2vec_model.train_w2v([sentence.strip().split(" ")
                                  for sentence in data])
    else:
        data2 = []
        for sentence in data:
            sentence = sentence.strip()
            if len(sentence) == 0:
                continue
            data2.append([])
            for char in sentence:
                if char == ' ':
                    continue
                data2[-1].append(char)
        word2vec_model.train_w2v(data2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='PyTorch model training')
    ap.add_argument('-t',
                    '--tag',
                    required=True,
                    type=str,
                    help='Seed for initializing training.')
    ap.add_argument('-s',
                    '--size',
                    required=True,
                    default=300,
                    type=int,
                    help='Batch size for dataset iterators')
    ap.add_argument('-w',
                    '--window',
                    default=5,
                    type=int,
                    help='Batch size for dataset iterators')

    args = vars(ap.parse_args())
    main(**args)

    '''
    python train_word2vec.py --tag=word --size=300
    python train_word2vec.py --tag=word --size=250
    python train_word2vec.py --tag=char --size=300
    python train_word2vec.py --tag=char --size=250
    
    '''
