# encoding:utf-8

from utils.data_utils import ClunerProcessor, ClunerDataLoader
from utils.log_utils import init_logger
from config.basic_config import configs
import os
from logging import Logger
import logging
import torch
import argparse
from torch import nn
from torch import optim
from schedulers import ReduceLRWDOnPlateau
from utils.progress_utils import ProgressBar
from utils.metric_utils import AverageMeter, SeqEntityScore
from models.bilstm_crf import BiLSTMCRFNERModel
from utils.runtime_utils import seed_everything


def load_and_cache_data(args, processor: ClunerProcessor = None,
                        data_type: str = 'train', logger: Logger = None):
    cache_file = os.path.join(args.data_dir, 'cached_{}_{}'.format(
        data_type, args.model_name
    ))
    if logger is None:
        logger = logging.getLogger()
    if os.path.exists(cache_file):
        logger.info("Loading features from cache file {}".format(cache_file))
        data = torch.load(cache_file)
    else:
        logger.info("Creating features from dataset {}".format(args.data_dir))
        if processor is None:
            processor = ClunerProcessor(args.data_dir)
        if data_type == 'train':
            data = processor.get_train_examples()
        elif data_type == 'dev':
            data = processor.get_dev_examples()
        logger.info("Saving features into cached file {}".format(cache_file))
        torch.save(data, cache_file)
    return data


def evaluate(args, model: nn.Module, processor: ClunerProcessor, logger: Logger):
    eval_dataset = load_and_cache_data(args, processor, data_type='dev')
    eval_dataloader = ClunerDataLoader(eval_dataset, batch_size=args.batch_size,
                                       shuffle=False,
                                       seed=args.seed,
                                       sort=False,
                                       vocab=processor.vocab,
                                       label2id=args.label2id)
    pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluation')
    metric = SeqEntityScore(args.id2label, markup=args.markup)
    eval_loss = AverageMeter()
    logger.info("start evaluate")
    model.eval()
    with torch.no_grad():
        for step in range(len(eval_dataloader)):
            batch = eval_dataloader[step]
            input_ids, input_mask, input_tags, input_lens = batch
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            input_tags = input_tags.to(args.device)
            features, loss = model.forward_loss(
                input_ids, input_mask, input_lens, input_tags)
            eval_loss.update(loss.item(), n=input_ids.size()[0])
            tags, _ = model.crf._obtain_labels(
                features, args.id2label, input_lens)
            input_tags = input_tags.detach().cpu().numpy()
            target = [input_[:len]
                      for input_, len in zip(input_tags, input_lens)]
            metric.update(target, tags)
            pbar(step=step)  # , info=metric.result()[1])
    logger.info("finish evaluate")
    eval_info, class_info = metric.result()
    eval_info = {f'eval_{key}': value for key, value in eval_info.items()}
    result = {'eval_loss': eval_loss.avg}
    result = dict(result, **eval_info)
    return result, class_info


def train(args, model: nn.Module, processor: ClunerProcessor, logger: Logger = None):
    if logger is None:
        global configs
        logger = init_logger("train", configs['log_dir'])
    train_dataset = load_and_cache_data(args, processor, logger=logger)
    train_loader = ClunerDataLoader(train_dataset, batch_size=args.batch_size,
                                    shuffle=args.shuffle,
                                    seed=args.seed,
                                    sort=args.sort,
                                    vocab=processor.vocab,
                                    label2id=args.label2id)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameters, lr=args.learning_rate)
    scheduler = ReduceLRWDOnPlateau(optimizer, mode='max', factor=0.6, patience=3,
                                    verbose=1, eps=1e-4, cooldown=0, min_lr=0, threshold=1e-6)

    best_f1 = 0
    for epoch in range(1, args.epochs):
        logger.info(f'Epoch [{epoch}/{args.epochs}] start training')
        pbar = ProgressBar(n_total=len(train_loader), desc='Training')
        train_loss = AverageMeter()
        model.train()
        assert model.training
        n_total = len(train_loader)
        for step in range(n_total):
            batch = train_loader[step]
            input_ids, input_mask, input_tags, input_lens = batch
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            input_tags = input_tags.to(args.device)
            features, loss = model.forward_loss(
                input_ids, input_mask, input_lens, input_tags)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            pbar(step=step, info={'loss': loss.item()})
            train_loss.update(loss.item(), n=1)
        logger.info(
            f'Epoch [{epoch}/{args.epochs}] finish training, loss:{train_loss.avg:.5f}')
        train_log = {'loss': train_loss.avg}
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
        eval_log, class_info = evaluate(args, model, processor, logger)
        logs = dict(**train_log, **eval_log)
        show_info = f'\nEpoch: {epoch}-' + \
            '-'.join([f' {key}: {value:.4f} ' for key, value in logs.items()])
        logger.info(show_info)
        scheduler.step(logs['eval_f1'])
        if logs['eval_f1'] > best_f1:
            logger.info(
                f"\n Epoch {epoch}: eval_f1 improved from {best_f1} to {logs['eval_f1']}")
            logger.info("save model to dick.")
            best_f1 = logs['eval_f1']
            if isinstance(model, nn.DataParallel):
                model_state_dict = model.modules.state_dict()
            else:
                model_state_dict = model.state_dict()
            state = {'epoch': epoch, 'model_name': args.model_name,
                     'state_dict': model_state_dict}
            model_path = os.path.join(args.output_model_path, "best_model.bin")
            torch.save(state, model_path)
        logger.info("Eval Entity Score")
        for key, value in class_info.items():
            info = f"Subject: {key} - Precision: {value['precision']} - Recall: {value['recall']} - F1: {value['f1']}"
            logger.info(info)


# TODO args 添加 data_dir
# TODO args 添加 model_name


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("--do_train", default=True, action='store_true')
    parse.add_argument('--model_name', default='bilstm_crf', type=str)

    parse.add_argument('--markup', default='bios', choices=['bio', 'bios'])
    parse.add_argument('--seed', default=2021, type=int)
    parse.add_argument('--batch_size', default=128, type=int)
    parse.add_argument('--sort', default=True, type=bool)
    parse.add_argument('--shuffle', default=True, type=bool)
    parse.add_argument('--learning_rate', default=0.001, type=float)
    parse.add_argument('--epochs', default=30, type=int)
    parse.add_argument("--gpu", default='0', type=str)
    parse.add_argument('--embedding_size', default=128, type=int)
    parse.add_argument('--hidden_size', default=384, type=int)
    parse.add_argument('--num_layers', default=2, type=int)
    parse.add_argument('--p_dropout', default=0.5, type=float)
    parse.add_argument("--grad_norm", default=5.0,
                       type=float, help="Max gradient norm.")
    parse.add_argument("--seed", default=2021, type=int)

    args = parse.parse_args()
    args.data_dir = configs['base_dir']
    args.label2id = configs['tag2id']
    args.id2label = configs['id2tag']
    args.output_dir = configs['output_dir']
    args.output_model_path = os.path.join(args.output_dir, args.model_name)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_model_path, exist_ok=True)
    if args.gpu != '':
        args.device = torch.device(f"cuda:{args.gpu}")
    else:
        args.device = torch.device("cpu")

    seed_everything(args.seed)

    processor = ClunerProcessor(configs['base_dir'])
    processor.get_vocab()
    logger = init_logger("train", configs['log_dir'])
    for label, idx in args.label2id.items():
        logger.info(f"label: {label} - id: {idx}")
    model = BiLSTMCRFNERModel(vocab_size=len(processor.vocab), embedding_size=args.embedding_size,
                              hidden_size=args.hidden_size, label2id=args.label2id,
                              device=args.device, p_dropout=args.p_dropout, num_layers=args.num_layers)
    model.to(args.device)
    if args.do_train:
        train(args, model, processor, logger)


if __name__ == "__main__":
    main()
