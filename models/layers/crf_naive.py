#! /usr/bin/bash python
# -*- encoding: utf-8 -*-
'''
@filename		: models/layers/crf_naive.py
@description	:
@created_time	: 2021/01/11 23:19:32
@author		: xinsen
version		: 1.0
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_scalar(var):
    return var.view(-1).detach().tolist()[0]


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def argmax_batch(vecs):
    _, idx = torch.max(vecs, 1)
    return idx


def log_sum_exp_batch(vecs):
    maxi = torch.max(vecs, 1)[0]
    maxi_bc = maxi[:, None].repeat(1, vecs.shape[1])
    recti_ = torch.log(torch.sum(torch.exp(vecs - maxi_bc), 1))
    return maxi + recti_


class CRF(nn.Module):
    def __init__(self, tagset_size, tag_dictionary, device, is_bert=None):
        super(CRF, self).__init__()

        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"
        if is_bert:
            self.START_TAG = "[CLS]"
            self.STOP_TAG = "[SEP]"
        self.tag_dictionary = tag_dictionary
        self.tagset_size = tagset_size
        self.device = device
        self.transitions = torch.randn(tagset_size, tagset_size)
        # self.transitions = torch.zeros(tagset_size, tagset_size)
        self.transitions.detach(
        )[self.tag_dictionary[self.START_TAG], :] = -10000
        self.transitions.detach(
        )[:, self.tag_dictionary[self.STOP_TAG]] = -10000
        self.transitions = self.transitions.to(device)
        self.transitions = nn.Parameter(self.transitions)
        # self.transitions[i,j] means transition score from tag_i to tag_j

    def _viterbi_decode(self, feats):
        backpointers = []
        backscores = []
        scores = []
        init_vvars = (torch.FloatTensor(1, self.tagset_size).to(
            self.device).fill_(-10000.0))
        init_vvars[0][self.tag_dictionary[self.START_TAG]] = 0
        # score, (1, self.tagset_size)
        forward_var = init_vvars

        for feat in feats:
            # next_tag_var, (num_tags, num_tags)
            # index [i,j] means the best score from start-tag to
            # i-th tag through j-th tag
            next_tag_var = (
                # index [i,j] means the current score of j-th tag
                forward_var.view(
                    1, -1).expand(self.tagset_size, self.tagset_size)
                # index [i,j] means the transition score from j-th tag to i-th tag
                + self.transitions
            )
            # bptrs_t, (num_tags)
            # demote p as bptrs_t[i], then p means the max score from start tag to
            # i-th tag must through p-th tag on current token
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            # viterbivars_t, (num_tags)
            # index i means the best score from start tag to i-th tag on current token
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]
            forward_var = viterbivars_t + feat
            backscores.append(forward_var)
            backpointers.append(bptrs_t)

        terminal_var = (
            forward_var
            + self.transitions[self.tag_dictionary[self.STOP_TAG]]
        )
        terminal_var.detach()[self.tag_dictionary[self.STOP_TAG]] = -10000.0
        terminal_var.detach()[self.tag_dictionary[self.START_TAG]] = -10000.0
        best_tag_id = argmax(terminal_var.unsqueeze(0))
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id.item())
        best_scores = []
        for backscore in backscores:
            softmax = F.softmax(backscore, dim=0)
            _, idx = torch.max(backscore, 0)
            prediction = idx.item()
            best_scores.append(softmax[prediction].item())
            scores.append([elem.item() for elem in softmax.flatten()])
        swap_best_path, swap_max_score = (
            best_path[0],
            scores[-1].index(max(scores[-1])),
        )
        scores[-1][swap_best_path], scores[-1][swap_max_score] = (
            scores[-1][swap_max_score],
            scores[-1][swap_best_path],
        )
        start = best_path.pop()
        assert start == self.tag_dictionary[self.START_TAG]
        best_path.reverse()
        return best_scores, best_path, scores

    def _forward_alg(self, feats, lens_):
        """
        feats: 发射分数, (batch_size, seq_length, num_tags)
        lens_: 长度张量, (batch_size)
        """
        init_alphas = torch.FloatTensor(self.tagset_size).fill_(-10000.0)
        init_alphas[self.tag_dictionary[self.START_TAG]] = 0.0

        forward_var = torch.zeros(
            feats.shape[0],
            feats.shape[1] + 1,
            feats.shape[2],
            dtype=torch.float,
            device=self.device,
        )
        forward_var[:, 0, :] = init_alphas[None, :].repeat(feats.shape[0], 1)
        # (batch_size, num_tags, num_tags)
        transitions = self.transitions.view(
            1, self.transitions.shape[0], self.transitions.shape[1]
        ).repeat(feats.shape[0], 1, 1)
        for i in range(feats.shape[1]):  # for i_th token in sequence
            # (batch_size, num_tags)
            emit_score = feats[:, i, :]
            # the index of [k,p,q] of
            # emit_score[:, :, None].repeat(1, 1, transitions.shape[2]) +
            # transitions
            # means
            # for k-th squence, the score from q-th tag to p-th tag
            # on i-th token
            # tag_var, (batch_size, num_tags, num_tags)
            # the index[k,p,q] of tag_var means
            # for k-th squence, the score of start tag to p-th tag through q-th tag
            # on i-th token
            tag_var = (
                # (batch_size, num_tags, num_tags)
                # [k,p,q] means for k-th squence, the emit score from p-th tag to i-th token
                emit_score[:, :, None].repeat(1, 1, transitions.shape[2])
                # (batch_size, num_tags, num_tags)
                # [k,p,q] means the transition score from q-th tag to p-th tag
                + transitions
                # [k,p,q] means for k-th squence, the previous score from start tag to q-th tag
                # on i-th token
                + forward_var[:, i, :][:, :, None]
                .repeat(1, 1, transitions.shape[2])
                .transpose(2, 1)
            )
            # max_tag_var, (batch_size, num_tags)
            # the index [k,p] of max_tag_var means
            # the score from start tag to p-th tag on i-th token
            # 此处使用 max_tag_var 的作用仅仅是用来做 log^sum^exp
            max_tag_var, _ = torch.max(tag_var, dim=2)
            tag_var = tag_var - max_tag_var[:, :, None].repeat(
                1, 1, transitions.shape[2]
            )
            agg_ = torch.log(torch.sum(torch.exp(tag_var), dim=2))
            cloned = forward_var.clone()
            cloned[:, i + 1, :] = max_tag_var + agg_
            forward_var = cloned
        # forward_var, (batch_size, num_tags)
        forward_var = forward_var[range(forward_var.shape[0]), lens_, :]
        terminal_var = forward_var + \
            self.transitions[self.tag_dictionary[self.STOP_TAG]
                             ][None, :].repeat(forward_var.shape[0], 1)
        alpha = log_sum_exp_batch(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags, lens_):
        """
            feats: 发射分数, (batch_size, seq_length, num_tags)
            tags: 标注张量, (batch_size, seq_length)
            lens_: 长度张量 (batch_size, 1) or (batch_size)
                                               probably this one
            方法非常巧妙, 使用双指针快速计算句子的分数
        """
        start = torch.LongTensor(
            [self.tag_dictionary[self.START_TAG]]).to(self.device)
        start = start[None, :].repeat(tags.shape[0], 1)  # (batch_size, 1)
        stop = torch.LongTensor(
            [self.tag_dictionary[self.STOP_TAG]]).to(self.device)
        stop = stop[None, :].repeat(tags.shape[0], 1)  # (batch_size, 1)
        # (batch_size, 1 + seq_length)
        pad_start_tags = torch.cat([start, tags], 1)
        # (batch_size, seq_length + 1)
        pad_stop_tags = torch.cat([tags, stop], 1)
        for i in range(len(lens_)):
            # make every padding tag be STOP_TAG
            pad_stop_tags[i, lens_[i]:] = self.tag_dictionary[self.STOP_TAG]
        # score, (batch_size)
        score = torch.FloatTensor(feats.shape[0]).to(self.device)
        for i in range(feats.shape[0]):
            # r, the index of valid part of sentence in current batch
            r = torch.LongTensor(range(lens_[i])).to(self.device)
            score[i] = torch.sum(
                self.transitions[
                    pad_stop_tags[i, : lens_[i] + 1],
                    pad_start_tags[i, : lens_[i] + 1]
                ]
            ) + torch.sum(feats[i, r, tags[i, : lens_[i]]])
        return score

    def _obtain_labels(self, feature, id2label, input_lens):
        tags = []
        all_tags = []
        for feats, length in zip(feature, input_lens):
            confidences, tag_seq, scores = self._viterbi_decode(feats[:length])
            tags.append([id2label[tag] for tag in tag_seq])
            all_tags.append(
                [[id2label[score_id] for score_id, score in enumerate(
                    score_dist)] for score_dist in scores
                 ])
        return tags, all_tags

    def calculate_loss(self, scores, tag_list, lengths):
        # 自顶向下分析
        # scores是发射分数矩阵(batch_size, seq_length, tagsize)
        # tag_list是标注序列张量(batch_size, seq_length, tagsize)
        # lengths是一个batch的长度
        return self._calculate_loss_old(scores, lengths, tag_list)

    def _calculate_loss_old(self, features, lengths, tags):
        """
        log likelihood = log \frac{exp \sum_k (emission_score(k)+transition_score(k))}
                                  {\sum_y exp \sum_k (emission_score(k)+transition_score(k))}
                       = \sum_k (emission_score(k)+transition_score(k)) -
                         log {\sum_y exp \sum_k (emission_score(k)+transition_score(k))}
        nll = log {\sum_y exp \sum_k (emission_score(k)+transition_score(k))} -
              \sum_k (emission_score(k)+transition_score(k))
            = forward_score - gold_score
        return the reduced average of nll
        """
        forward_score = self._forward_alg(features, lengths)
        gold_score = self._score_sentence(features, tags, lengths)
        score = forward_score - gold_score
        return score.mean()


if __name__ == "__main__":
    a = torch.arange(6).view(3, 2)
    print(a)
    print(to_scalar(a))
