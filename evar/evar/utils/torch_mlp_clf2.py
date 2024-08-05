"""
TorchMLPClassifier ver.2

A PyToch based Multi-Layer Perceptron Classifier, mostly compatible interface with scikit-learn.
Running on GPU by default.

Major changes from ver.1:
    - Detailed assessment is possible: score() function will also return a pandas data frame
    that describes the per-class result.
    - Mixup is available: it might be helpful if you suffer overfitting by the scarcity of
    training samples.

Disclimer:
    NOT FULLY COMPATIBLE w/ scikit-learn.

References:
    - https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    - https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/neural_network/_multilayer_perceptron.py
"""

import copy
import logging
import os
import random
import time
from typing import Mapping, Sequence

import mir_eval
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.datasets
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
import sklearn.metrics as skmetrics


log = logging.getLogger(__name__)


class Mixup(nn.Module):

    def __init__(self, mixup_alpha=0.1):
        super(Mixup, self).__init__()
        self.mixup_alpha = mixup_alpha
        logging.info(f' using mixup with alpha={mixup_alpha}')

    def get_lambda(self, batch_size, device):
        lambdas = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size)
        self.lambdas = torch.tensor(lambdas).to(torch.float).to(device)
        self.counter_indexes = np.random.permutation(batch_size)

    def forward(self, x_and_y):
        def do_mixup(x, mixup_lambda):
            x = x.transpose(0, -1)
            out = x * self.lambdas + x[..., self.counter_indexes] * (1.0 - self.lambdas)
            return out.transpose(0, -1)

        self.get_lambda(len(x_and_y[0]), x_and_y[0].device)
        x_or_y = [do_mixup(z, self.lambdas) for z in x_and_y]
        return x_or_y


def seed_everything(seed=42):
    if seed is None: return
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_array_like(item):
    """Check if item is an array-like object."""
    return isinstance(item, (list, set, tuple, np.ndarray))


def all_same_classes(y_a, y_b, delimiter=None):
    """Test if all classes in y_a is also in y_b or not.
    If y_a is a single dimension array, test as single labeled.
    If y_a is a two dimension array, test as multi-labeled.

    Args:
        y_a: One list of labels.
        y_b: Another list of labels.
        delimiter: Set a character if multi-label text is given.

    Returns:
        True or False.
    """
    if is_array_like(y_a[0]):
        # binary matrix multi-label table, test that class existance is the same.
        y_a, y_b = y_a.sum(axis=0), y_b.sum(axis=0)
        classes_a, classes_b = y_a > 0, y_b > 0
        return np.all(classes_a == classes_b)

    # test: classes contained in both array is consistent.
    if delimiter is not None:
        y_a = flatten_list([y.split(delimiter) for y in y_a])
        y_b = flatten_list([y.split(delimiter) for y in y_b])
    classes_a, classes_b = list(set(y_a)), list(set(y_b))
    return len(classes_a) == len(classes_b)


def train_test_sure_split(X, y, n_attempt=100, return_last=True, **kwargs):
    """Variant of train_test_split that makes validation for sure.
    Returned y_test should contain all class samples at least one.
    Simply try train_test_split repeatedly until the result satisfies this condition.

    Args:
        n_attempt: Number of attempts to satisfy class coverage.
        return_last: Return last attempt results if all attempts didn't satisfy.

    Returns:
        X_train, X_test, y_train, y_test if satisfied;
        or None, None, None, None.
    """

    for i in range(n_attempt):
        X_trn, X_val, y_trn, y_val = train_test_split(X, y, **kwargs)
        if all_same_classes(y, y_val):
            return X_trn, X_val, y_trn, y_val
    if return_last:
        return X_trn, X_val, y_trn, y_val
    return None, None, None, None


class EarlyStopping:
    def __init__(self, target='acc', objective='max', patience=10, enable=True):
        self.crit_targ = target
        self.crit_obj = objective
        self.patience = patience
        self.enable = enable
        self.stopped_epoch = 0
        self.wait = 0
        self.best_value = 0 if objective == 'max' else 1e15
        self.best_epoch = None
        self.best_weights = None
        self.best_metrics = None

    def on_epoch_end(self, epoch, model, val_metrics):
        status = False
        condition = (val_metrics[self.crit_targ] >= self.best_value
                     if self.crit_obj == 'max' else
                     val_metrics[self.crit_targ] <= self.best_value)
        if condition:
            self.best_epoch = epoch
            self.best_weights = copy.deepcopy(model.state_dict())
            self.best_metrics = val_metrics
            self.best_value = val_metrics[self.crit_targ]
            self.wait = 1
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch + 1
                status = self.enable
            self.wait += 1
        return status


def _validate(device, model, dl, criterion, return_values=True):
    model.eval()
    all_targets, all_preds = [], []
    with torch.no_grad():
        val_loss = 0.0
        for inputs, targets in dl:
            all_targets.extend(targets.numpy())
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets) * inputs.size(0)
            if targets.dim() == 1:
                outputs = outputs.softmax(-1).argmax(-1)
            elif targets.dim() == 2:
                outputs = outputs.sigmoid()
            all_preds.extend(outputs.detach().cpu().numpy())
        val_loss /= len(dl)
    if return_values:
        return val_loss, np.array(all_targets), np.array(all_preds)
    return val_loss


def _train(device, model, dl, criterion, optimizer, scheduler=None, mixup=None):
    model.train()
    train_loss = 0.0
    for inputs, labels in dl:
        inputs = inputs.to(device)
        labels = labels.to(device)
        if mixup is not None:
            inputs, labels = mixup.forward([inputs, labels])
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    if scheduler:
        scheduler.step()
    train_loss /= len(dl)
    return train_loss


def loss_nll_with_logits(logits, gts):
    logprobs = F.log_softmax(logits, dim=-1)
    loss = -torch.mean(gts * logprobs)
    return loss


def loss_bce_with_logits(logits, gts):
    preds = torch.sigmoid(logits)
    return F.binary_cross_entropy(preds, gts)


def eval_map(y_score, y_true, classes):
    average_precision = skmetrics.average_precision_score(y_true, y_score, average=None)
    if classes is None:
        return average_precision.mean(), None
    return average_precision.mean(), pd.DataFrame({'ap': average_precision, 'class': classes})


def eval_roc(y_score, y_true, classes):
    auc = skmetrics.roc_auc_score(y_true, y_score, average=None)
    return auc.mean(), pd.DataFrame({'auc': auc, 'class': classes}) if classes is not None else None


def eval_acc(preds, labels, classes):
    preds = np.argmax(preds, axis=-1)
    labels = np.argmax(labels, axis=-1)
    accuracy = labels == preds
    if classes is None:
        return accuracy.mean(), pd.DataFrame({"true": labels, "pred": preds})

    def class_name(indexed): return [classes[l] for l in indexed]

    return accuracy.mean(), pd.DataFrame({'true': class_name(labels), 'pred': class_name(preds)})


def eval_mirex(preds, labels, classes=None):
    preds = np.argmax(preds, axis=-1)

    keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    modes = ['major', "minor"]

    score = 0

    ref_keys, est_keys = [], []

    for pred, label in zip(preds, labels):
        mp, kp = divmod(pred, 12)
        key_pred = keys[kp] + ' ' + modes[mp]

        valid_scores = []
        for l in np.where(label)[0]:
            ml, kl = divmod(l, 12)
            key_label = keys[kl] + ' ' + modes[ml]
            valid_scores.append(mir_eval.key.weighted_score(key_label, key_pred))

        score += max(valid_scores)
        ref_keys.append(key_label)
        est_keys.append(key_pred)

    return score / len(preds), pd.DataFrame({"true": est_keys, "pred": ref_keys})


def eval_acc1(preds, labels, classes=None):
    shift = classes.shape[-1] - preds.shape[-1] - 1

    preds = np.argmax(preds, axis=-1) + shift
    labels = np.argmax(labels, axis=-1) + shift

    ref_tempo = 1.04 ** (2 * labels)
    est_tempo = 1.04 ** (2 * preds)

    correct = np.abs(ref_tempo - est_tempo) <= 0.04 * ref_tempo

    return np.mean(correct), pd.DataFrame({"true": ref_tempo, "pred": est_tempo})


def eval_acc2(preds, labels, classes=None):
    shift = classes.shape[-1] - preds.shape[-1] - 1

    preds = np.argmax(preds, axis=-1) + shift
    labels = np.argmax(labels, axis=-1) + shift

    ref_tempo = 1.04 ** (2 * labels)
    est_tempo = 1.04 ** (2 * preds)

    correct = np.zeros(labels.shape, dtype=bool)

    for m in [1/3, 1/2, 1, 2, 3]:
        ref = m * ref_tempo
        correct |= np.abs(ref - est_tempo) <= 0.04 * ref

    return np.mean(correct), None

def _calc_metric(metrics: Sequence[str], targets, preds, classes):
    if isinstance(metrics, str):
        metrics = [metrics]
    results = {}
    df = None
    for metric in metrics:
        compute_metric = eval(f"eval_{metric.lower()}")
        res, df = compute_metric(preds, targets, classes)
        results[metric] = res

    return results, df


def _train_model(device, model, criterion, optimizer, scheduler, trn_dl, val_dl, metric="acc",
                 num_epochs=200, seed=None, patience=10, stop_metric=None,
                 early_stopping=False, mixup=True):
    seed_everything(seed)
    stop_metric = metric if stop_metric is None else stop_metric
    stop_objective = 'min' if stop_metric == 'loss' else 'max'
    early_stopper = EarlyStopping(patience=patience, target=stop_metric, objective=stop_objective,
                                  enable=early_stopping)
    mixup = Mixup() if mixup else None
    since = time.time()

    log_freq = 1 if log.getEffectiveLevel() <= logging.DEBUG else 50

    for epoch in range(num_epochs):
        # train
        trn_loss = _train(device, model, trn_dl, criterion, optimizer, scheduler, mixup)
        # validate, calculate metrics
        val_loss, val_targets, val_preds = _validate(device, model, val_dl, criterion)
        val_metrics, _ = _calc_metric(metric, val_targets, val_preds, classes=None)
        val_metrics['loss'] = val_loss
        # print log
        cur_lr = optimizer.param_groups[0]["lr"]

        if epoch % log_freq == 0:
            log.info(f'epoch {epoch + 1:04d}/{num_epochs}: lr: {cur_lr:.7f}: loss={trn_loss:.6f} '
                     + ' '.join([f'val_{n}={v:.7f}' for n, v in val_metrics.items()]))
        # early stopping
        if early_stopper.on_epoch_end(epoch, model, val_metrics):
            break

    time_elapsed = time.time() - since
    log.debug(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    for n, v in early_stopper.best_metrics.items():
        log.debug(f'Best val_{n}@{early_stopper.best_epoch + 1} = {v}')

    # load best model weights
    model.load_state_dict(early_stopper.best_weights)
    return model, early_stopper.best_epoch, early_stopper.best_metrics


class MLP(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        sizes = [input_size] + list(hidden_sizes) + [output_size]
        layers = []
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
        self.mlp = nn.Sequential(*layers[:-2])

    def forward(self, x):
        out = self.mlp(x)
        return out


class TorchMLPClassifier2:

    def __init__(self, hidden_layer_sizes=(100,), activation="relu", *,
                 solver='adam', alpha=1e-8,  # alpha=0.0001 --- too big for this implementation
                 batch_size: int | str = 'auto', learning_rate="constant",
                 learning_rate_init=0.001, power_t=0.5, max_iter=200,
                 shuffle=True, random_state=None, tol=1e-4,
                 warm_start=False, momentum=0.9,
                 nesterovs_momentum=True, early_stopping=False,
                 validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, n_iter_no_change=10, max_fun=15000,
                 # Extra options
                 standard_scaler=True, mixup=False):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.loss = 'log_loss'
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.max_fun = max_fun
        self.standard_scaler = StandardScaler() if standard_scaler else None
        self.mixup = mixup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.oh_enc = None

        # placeholders
        self.model = None
        self.optimizer = None
        self.criterion = None

    def switch_regime(self, y, y_val, metrics: Sequence[str] = None):
        if y.ndim == 2:  # multi label
            n_class = y.shape[1]
            return metrics or ['mAP'], loss_bce_with_logits, n_class, y, y_val

        if y.ndim == 1:  # classification -> one-hot encoding
            if self.oh_enc is None:
                self.oh_enc = LabelBinarizer()
                self.oh_enc.fit(y)
            y = self.oh_enc.transform(y)
            if y_val is not None:
                y_val = self.oh_enc.transform(y_val)
            n_class = len(self.oh_enc.classes_)
            return metrics or ['acc'], loss_nll_with_logits, n_class, y, y_val

        raise Exception(f'Unsupported shape of y: {y.shape}')

    def fit(self, X, y, X_val=None, y_val=None, val_idxs=None):
        metrics, criterion, n_class, y, y_val = self.switch_regime(y, y_val)
        metric = metrics[0]

        n_samples = len(X)
        bs = min(200, n_samples) if self.batch_size == 'auto' else self.batch_size
        train_kwargs = {'batch_size': bs, 'drop_last': False, 'shuffle': self.shuffle}
        test_kwargs = {'batch_size': bs, 'drop_last': False, 'shuffle': False}

        if self.standard_scaler:
            self.standard_scaler.fit(X)
            X = self.standard_scaler.transform(X)
            if X_val is not None:
                X_val = self.standard_scaler.transform(X_val)

        if X_val is not None:
            Xtrn, Xval, ytrn, yval = X, X_val, y, y_val
        elif val_idxs is None:
            Xtrn, Xval, ytrn, yval = train_test_sure_split(X, y, test_size=self.validation_fraction,
                                                           random_state=self.random_state)
        else:
            mask = np.array([i in val_idxs for i in range(len(X))])
            Xtrn, Xval, ytrn, yval = X[~mask], X[mask], y[~mask], y[mask]

        Xtrn, Xval, ytrn, yval = [torch.from_numpy(_x) for _x in [Xtrn, Xval, ytrn, yval]]

        log.debug(f' stats|train: mean={Xtrn.mean():.4f}, std={Xtrn.std():.4f}')
        log.debug(f' stats|valid: mean={Xval.mean():.4f}, std={Xval.std():.4f}')

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtrn, ytrn), **train_kwargs)
        eval_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xval, yval), **test_kwargs)

        model = MLP(input_size=X.shape[-1], hidden_sizes=self.hidden_layer_sizes, output_size=n_class)
        self.model = model.to(self.device)
        self.optimizer = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}[self.solver](self.model.parameters(),
                                                                                             lr=self.learning_rate_init,
                                                                                             betas=(
                                                                                             self.beta_1, self.beta_2),
                                                                                             eps=self.epsilon,
                                                                                             weight_decay=self.alpha)
        self.criterion = criterion
        log.debug(f'Training model: {model}')
        log.debug(
            f'Details - metric: {metric}, loss: {criterion}, optimizer: {self.optimizer}, n_class: {n_class}')

        return _train_model(self.device, self.model, self.criterion, self.optimizer, None, train_loader, eval_loader,
                            metric=metric,
                            num_epochs=self.max_iter, seed=self.random_state, patience=self.n_iter_no_change,
                            early_stopping=self.early_stopping, mixup=self.mixup)

    def score(self, test_X, test_y, classes=None, metrics: Sequence[str] | None = None):
        """
            classes: Display classes.
        """
        metrics, criterion, n_class, test_y, _ = self.switch_regime(test_y, None, metrics=metrics)

        bs = 256 if self.batch_size == 'auto' else self.batch_size
        test_kwargs = {'batch_size': bs, 'drop_last': False, 'shuffle': False}

        if self.standard_scaler:
            test_X = self.standard_scaler.transform(test_X)
        logging.info(f' stats|test: mean={test_X.mean():.4f}, std={test_X.std():.4f}')

        Xval, yval = torch.Tensor(test_X), torch.Tensor(test_y)
        eval_loader = DataLoader(TensorDataset(Xval, yval), **test_kwargs)

        val_loss, targets, preds = _validate(self.device, self.model, eval_loader, self.criterion)
        results, df = _calc_metric(metrics, targets, preds, classes=classes)
        return results, df

    def predict(self, X, multi_label_n_class=None):
        bs = 256 if self.batch_size.lower() == 'auto' else self.batch_size
        test_kwargs = {'batch_size': bs, 'drop_last': False, 'shuffle': False}
        if self.standard_scaler:
            X = self.standard_scaler.transform(X)
        X = torch.Tensor(X)
        y = (torch.zeros((len(X)), dtype=torch.int) if multi_label_n_class is None else
             torch.zeros((len(X), multi_label_n_class), dtype=torch.float))
        eval_loader = DataLoader(torch.utils.data.TensorDataset(X, y), **test_kwargs)

        val_loss, targets, preds = _validate(self.device, self.model, eval_loader, self.criterion)
        return preds
