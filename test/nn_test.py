from models.nn.bilstm import BiLSTMModel
from utils.log_utils import init_logger
from utils.data_utils import DataTransformer, NERDataLoader
from config.basic_config import configs
import os
from torch.nn import functional as F
import torch

if __name__ == "__main__":
    logger = init_logger(
        log_name='main', log_dir=configs['log_dir'])
    transformer = DataTransformer(os.path.join(
        configs['embedding_dir'], 'char_word2vec_size200_win5.txt'), logger,
        all_data_path=configs['all_data_path'], tag2id=configs['tag2id'],
        max_length=configs['max_length'])
    # 获取 word2id 和 embedding_matrix
    logger.info("loading word2id and embedding matrix")
    word2id = transformer.word2id
    embedding_matrix = transformer.embedding_matrix
    # 获取 dataloader
    ner_base_loader = NERDataLoader(transformer, configs['device'],
                                    configs['max_length'], logger=logger)
    logger.info("training data loader")
    train_dataloader = ner_base_loader.load_data(
        configs['ptrain_x_path'], configs['ptrain_y_path'], True,
        configs['batch_size'], configs['num_workers'], shuffle=True)
    logger.info("validation data loader")
    val_dataloader = ner_base_loader.load_data(
        configs['pval_x_path'], configs['pval_y_path'], True,
        configs['batch_size'], configs['num_workers'], shuffle=True)
    logger.info("test data loader")
    test_dataloader = ner_base_loader.load_data(
        configs['ptest_x_path'], None, True,
        configs['batch_size'], configs['num_workers'], shuffle=False)
    model = BiLSTMModel(model_config=configs,
                        embedding_dim=embedding_matrix.shape[1],
                        num_classes=configs['num_classes'],
                        vocab_size=embedding_matrix.shape[0],
                        embedding_weight=embedding_matrix)
    for input_words, tags, mask in train_dataloader:
        output_logits = model(input_words, mask)
        # print(output_logits)
        print(output_logits.shape)
        # F.cross_entropy()
        output_logits = F.log_softmax(output_logits, dim=2)
        print(output_logits)
        print(output_logits.shape)
        loss = F.nll_loss(output_logits.view(-1, output_logits.size()[-1]), tags.view(-1), reduction= 'none')
        # print(loss)
        loss = loss.view(mask.size()[0], mask.size()[1])
        loss = torch.mul(loss, mask)
        print(loss.shape)
        print(torch.mean(loss))
        break