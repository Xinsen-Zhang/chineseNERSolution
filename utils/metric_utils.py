# encoding:utf-8
# import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Optional

from collections import Counter
from .ner_utils import get_entities


class Accuracy(object):
    def __init__(self):
        pass

    def __call__(self, predict_tags: torch.LongTensor,
                 actual_tags: torch.LongTensor,
                 masks: torch.LongTensor):
        predict_sequence = []
        actual_sequence = []
        predict_tags = predict_tags.detach().cpu().from_numpy().tolist()
        actual_tags = actual_tags.detach().cpu().from_numpy().tolist()
        length = torch.sum(masks.long(), dim=-
                           1).detach().cpu().from_numpy().tolist()
        batch_size = len(predict_tags)
        for i in range(batch_size):
            for j in range(length[i]):
                predict_sequence.append(predict_tags[i][j])
                actual_sequence.append(actual_tags[i][j])
        return accuracy_score(y_pred=predict_sequence, y_true=actual_sequence)


class F1PrecisionRecall(object):
    def __init__(self, num_classes: Optional[int] = None):
        self.num_classes = num_classes
        if num_classes is None:
            pass
        else:
            self.labels = [i for i in range(self.num_classes)]
        self.accuracy = Accuracy()

    def __call__(self, predict_tags: torch.LongTensor,
                 actual_tags: torch.LongTensor,
                 masks: torch.LongTensor):
        predict_sequence = []
        actual_sequence = []
        predict_tags = predict_tags.detach().cpu().numpy().tolist()
        actual_tags = actual_tags.detach().cpu().numpy().tolist()
        length = torch.sum(masks.long(), dim=-
                           1).detach().cpu().numpy().tolist()
        batch_size = len(predict_tags)
        for i in range(batch_size):
            for j in range(length[i]):
                predict_sequence.append(predict_tags[i][j])
                actual_sequence.append(actual_tags[i][j])
        if self.num_classes is not None:
            f1 = f1_score(y_pred=predict_sequence, y_true=actual_sequence,
                          labels=self.labels, average='macro')
            p = precision_score(y_pred=predict_sequence, y_true=actual_sequence,
                                labels=self.labels, average='macro')
            r = recall_score(y_pred=predict_sequence, y_true=actual_sequence,
                             labels=self.labels, average='macro')
        else:
            f1 = f1_score(y_pred=predict_sequence, y_true=actual_sequence)
            p = precision_score(y_pred=predict_sequence,
                                y_true=actual_sequence)
            r = recall_score(y_pred=predict_sequence, y_true=actual_sequence)
        acc = accuracy_score(y_pred=predict_sequence, y_true=actual_sequence)
        return acc, f1, p, r


class SeqEntityScore(object):
    def __init__(self, id2label, markup='bios'):
        self.id2label = id2label
        self.markup = markup
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else right / origin
        precision = 0 if found == 0 else right / found
        f1 = 0.0 if recall + \
            precision == 0 else (2*recall*precision)/(precision+recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter([x[0] for x in self.origins])
        found_counter = Counter(x[0] for x in self.founds)
        right_counter = Counter(x[0] for x in self.rights)
        for category, count in origin_counter.items():
            origin = count
            found = found_counter.get(category, 0)
            right = right_counter.get(category, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[category] = {
                "recall": recall,
                "precision": precision,
                "f1": f1
            }
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {
            'recall': recall,
            'precision': precision,
            'f1': f1
        }, class_info

    def update(self, label_paths, pred_paths):
        """
        labels_paths = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        pred_paths = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        """
        for label_path, pred_path in zip(label_paths, pred_paths):
            label_entities = get_entities(
                label_path, self.id2label, self.markup)
            pred_entities = get_entities(pred_path, self.id2label, self.markup)
            self.origins.extend(label_entities)
            self.founds.extend(pred_entities)
            self.rights.extend([
                pred_entity for pred_entity in pred_entities if pred_entity in label_entities
            ])


class AverageMeter(object):
    def __init__(self):
        self._reset()

    def _reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0.0

    def update(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        # self.avg = self.sum /
