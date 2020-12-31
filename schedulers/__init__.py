# encoding:utf-8

import math
import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLRF
import warnings

EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)


__all__ = [
    'CustomDecayLR',
    'BertLR',
    'CyclicLR',
    'ReduceLROnPlateau',
    'ReduceLRWDOnPlateau',
    'CosineLRWithRestarts'
]


class CustomDecayLR(object):
    '''
    自定义学习率变化机制
        Example:
        >>> scheduler = CustomDecayLR(optimizer)
        >>> for epoch in range(100):
        >>>     scheduler.epoch_step()
        >>>     train(...)
        >>>         ...
        >>>         optimizer.zero_grad()
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>     validate(...)
    '''

    def __init__(self, optimizer: Optimizer, lr):
        self.optimizer = optimizer
        self.lr = lr

    def epoch_step(self, epoch):
        lr = self.lr
        if epoch > 12:
            lr = lr / 1000
        elif epoch > 8:
            lr = lr / 100
        elif epoch > 4:
            lr = lr / 10
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class BertLR(object):
    """
    Bert模型内定的学习率变化机制
    """

    def __init__(self, optimizer: Optimizer, learning_rate: float, t_total, warmup):
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.t_total = t_total
        self.warmup = warmup

    # 线性预热方式
    def warmup_linear(self, x, warmup=0.002):
        if x < warmup:
            return x / warmup
        return 1.0 - x

    def batch_step(self, traning_step):
        lr_this_step = self.learning_rate * \
            self.warmup_linear(traning_step/self.t_total, self.warmup)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_this_step


class CyclicLR:
    """
    Cyclic learning rate for traning nuural network
    """

    def __init__(self, optimizer: Optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode="triangular", gamma=1.0,
                 scale_fn=None, scale_mode='cyclic', last_batch_iteration=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))

        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(self.optimizer.param_groups):
                raise ValueError("expected {} base_lr, but got {}".format(
                    len(self.optimizer.param_groups), len(base_lr)
                ))
            self.base_lr = [lr for lr in base_lr]
        else:
            self.base_lr = [base_lr]*len(self.optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(self.optimizer.param_groups):
                raise ValueError("expected {} max_lr, but got {}".format(
                    len(self.optimizer.param_groups), len(max_lr)
                ))
            else:
                self.max_lr = [lr for lr in max_lr]
        else:
            self.max_lr = [max_lr] * len(self.optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] and scale_fn is None:
            raise ValueError("mode is invalid and scale_fn is None")

        self.gamma = gamma
        self.mode = mode

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iteration'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration+1)
        self.last_batch_iteration = last_batch_iteration

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x-1))

    def _exp_range_scale_fn(self, x):
        return self.gamma ** (x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1+self.last_batch_iteration/(2*step_size))
        x = np.abs(self.last_batch_iteration/step_size-2*cycle+1)

    def batch_step(self, batch_iteration=None):
        pass


class ReduceLROnPlateau(object):

    def __init__(self, optimizer: Optimizer,
                 mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):

        if factor > 1.0:
            raise ValueError("Factor should be < 1.0")
        self.factor = factor

        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(
                type(optimizer).__name__)
            )
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(mode=self.mode, threshold=self.threshold,
                             threshold_mode=self.threshold_mode)
        self._reset()

    def _reset(self):
        """
        Resets num_bad_epochs counter and cooldown counter.
        """
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold,
                             threshold_mode=self.threshold_mode)

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode = 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1 - self.threshold
            return a < best * rel_epsilon
        elif self.mode = 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold
        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = 1+self.threshold
            return a > best * rel_epsilon
        else:
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + " is unknown!")
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' +
                             threshold_mode + " is unknown!")

        if mode == 'min':
            self.mode_worse = math.inf
        else:
            self.mode_worse = -math.inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate of group {} to {:.4e}'.format(
                        epoch, i, new_lr
                    ))

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch+1
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
            pass
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class ReduceLRWDOnPlateau(ReduceLROnPlateau):

    def step(self, metrics, epoch):
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch+1
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
            pass
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0

        if self.num_bad_epochs > self.patience:
            self._reduce_lr_wd(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _reduce_lr_wd(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(self.min_lrs[i], old_lr * self.factor)
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate of group {} to {:.4e}'.format(
                        epoch, i, new_lr
                    ))
            old_weight_decay = float(param_group['weight_decay'])
            if old_weight_decay != 0:
                continue
            # new_weight_decay = max(old_weight_decay * self.factor, self.min_lrs[i])
            # TODO 此处存疑
            new_weight_decay = old_weight_decay * self.factor
            if old_weight_decay - new_weight_decay > self.eps:
                param_group['weight_decay'] = new_weight_decay
                if self.verbose:
                    print('Epoch {:5d}: reducing weight decay of group {} to {:.4e}'.format(
                        epoch, i, new_weight_decay
                    ))
