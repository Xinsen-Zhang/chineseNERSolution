from models.nn.bilstm_crf import BiLSTMCRFModel
from utils.log_utils import init_logger
from utils.data_utils import DataTransformer, NERDataLoader
from config.basic_config import configs
# from models.losses.cross_entropy import CrossEntropyLossWithMask
import os
from torch import optim
from utils.metric_utils import F1PrecisionRecall
from warnings import filterwarnings
filterwarnings("ignore")

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
    model = BiLSTMCRFModel(model_config=configs,
                           embedding_dim=embedding_matrix.shape[1],
                           num_classes=configs['num_classes'],
                           vocab_size=embedding_matrix.shape[0],
                           embedding_weight=embedding_matrix).to(configs['device'])
    # criterion = CrossEntropyLossWithMask().to(configs['device'])
    optimizer = optim.Adam(params=model.parameters(), lr=configs['learning_rate'],
                           weight_decay=configs['weight_decay'])
    metric = F1PrecisionRecall(configs['num_classes'])
    for epoch_index in range(100):
        total_losses = 0.0
        for train_index, (input_words, tags, mask) in enumerate(train_dataloader):
            model.train()
            optimizer.zero_grad()
            input_words = input_words.to(configs['device'])
            tags = tags.to(configs['device'])
            mask = mask.to(configs['device'])
            loss = model(input_words, mask, tags)
            # loss = criterion(output_logits, tags, mask)
            loss.backward()
            optimizer.step()
            total_losses += loss.detach().cpu().item()
            predicted_tags = model.inference(input_words, mask)
            acc, f1, precision, recall = metric(predicted_tags, tags, mask)
            if train_index % 10 == 0 or train_index + 1 == len(train_dataloader):
                logger.info("train_index: {}-[{}/{}] loss: {:.3f}, acc: {:.3f}%, f1: {:.3f}, precision:{:.3f}, recall: {:.3f}".
                            format(epoch_index,
                                   train_index,
                                   len(train_dataloader), total_losses /
                                   (train_index+1),
                                   acc*100, f1,
                                   precision,
                                   recall))
            # break
        # break
