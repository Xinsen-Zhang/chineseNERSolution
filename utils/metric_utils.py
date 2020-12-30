# encoding:utf-8
# import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Optional


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
