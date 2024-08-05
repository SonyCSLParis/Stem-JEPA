"""Linear evaluation runner.

NAME
    lineareval.py

SYNOPSIS
    lineareval.py CONFIG_FILE TASK <flags>

POSITIONAL ARGUMENTS
    CONFIG_FILE
    TASK

FLAGS
    --options=OPTIONS
        Default: Nothing to change.
    --seed=SEED
        Random seed used to train and test a linear (or MLP) model.
        Default: 42
    --lr=LR
        Default: None
    --hidden=HIDDEN
        Defines a small model to solve the task.
        `()` means linear evaluation, and `(512,)` means one hidden layer with 512 units for example.
        Default: (), i.e., linear evaluation
    --standard_scaler=STANDARD_SCALER
        Default: True
    --mixup=MIXUP
        Default: False
    --epochs=EPOCHS
        Default: None
    --early_stop_epochs=EARLY_STOP_EPOCHS
        Default: None
    --step=STEP
        Default: '1pass'
"""

import json
import logging
import os
from pathlib import Path
from typing import Sequence, Tuple

import fire
import numpy as np
import plotly.express as px
from tqdm import tqdm

import torch

import evar.ar_lightning
from evar.ds_tasks import get_metrics


log = logging.getLogger(__name__)


from evar.common import complete_cfg, kwarg_cfg, app_setup_logger, setup_dir, RESULT_DIR, LOG_DIR
from evar.data import create_dataloader
from evar.ds_tasks import get_defs
from evar.utils import append_to_csv, hash_text, load_yaml_config, seed_everything, confusion_matrix
from evar.utils.torch_mlp_clf2 import TorchMLPClassifier2

torch.backends.cudnn.benchmark = True
# Workaround for "RuntimeError: Too many open files. Communication with the workers is no longer possible."
torch.multiprocessing.set_sharing_strategy('file_system')


def get_cache_info(split, _id, fold, label: str = "label"):
    # store in local compute node if within a job
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is not None:
        cache_dir = Path(f'/local/job/{job_id}/cache')
    else:
        cache_dir = Path('work/cache')
    
    cache_file = cache_dir / f'embs-{_id}-{split}-{fold}.npy'
    cache_gt_file = cache_dir / f'embs-{_id}-{split}-{fold}-{label}.npy'
    return cache_file, cache_gt_file


def to_embeddings(emb_ar: torch.nn.Module,
                  data_loader,
                  device,
                  _id=None,
                  fold: int = 1,
                  split: str = "train",
                  label: str = "label",
                  cache: bool = False,
                  batch_size: int = 64) -> Tuple[np.ndarray | None, np.ndarray | None, int]:
    cache_file, cache_gt_file = get_cache_info(split, _id, fold, label)

    cached_embs, cached_gts = False, False
    embs, gts = [], []
    if cache_file.exists():
        log.info(f' using cached embeddings: {cache_file.stem}')
        cached_embs = True
        embs = np.load(cache_file)

    if cache_gt_file.exists():
        log.info(f' using cached ground-truths: {cache_gt_file.stem}')
        cached_gts = True
        gts = np.load(cache_gt_file)

    if cached_embs and cached_gts:
        return embs, gts, batch_size

    if emb_ar is not None:
        emb_ar.eval()

    while True:
        try:
            for X, y in tqdm(data_loader(batch_size=batch_size),
                             desc=f'Getting {_id} {split} embeddings...',
                             leave=False):
                if not cached_embs:
                    with torch.no_grad():
                        embs.append(emb_ar(X.to(device)).detach().cpu())
                gts.append(y)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if batch_size == 1:
                raise e

            batch_size //= 2
            log.info(f"Caught OOM error, retrying with batch size={batch_size}.")
            embs, gts = [], []
            torch.cuda.empty_cache()
            continue

        break

    if len(embs) == 0:  # empty dataloader, e.g. because of cross-validation
        return None, None, batch_size

    cache_file.parent.mkdir(exist_ok=True, parents=True)
    if not cached_embs:
        embs = torch.cat(embs, dim=0).numpy()
        if cache:
            np.save(cache_file, embs)

    gts = torch.cat(gts, dim=0).numpy()
    if cache and not cached_gts:
        np.save(cache_gt_file, gts)

    return embs, gts, batch_size


def _one_linear_eval(X, y, X_val, y_val, X_test, y_test,
                     batch_size: int, hidden_sizes,
                     epochs, early_stop_epochs,
                     lr, classes,
                     metrics: Sequence[str] | None,
                     standard_scaler, mixup):
    log.info(f'🚀 Started {"Linear" if hidden_sizes == () else f"MLP with {hidden_sizes}"} evaluation:')
    clf = TorchMLPClassifier2(hidden_layer_sizes=hidden_sizes, max_iter=epochs, learning_rate_init=lr,
                              batch_size=batch_size,
                              early_stopping=early_stop_epochs > 0, n_iter_no_change=early_stop_epochs,
                              standard_scaler=standard_scaler, mixup=mixup)
    clf.fit(X, y, X_val=X_val, y_val=y_val)
    score, df = clf.score(X_test, y_test, classes, metrics=metrics)
    return score, df


def make_cfg(config_file: str,
             task: str,
             options,
             extras={},
             cancel_aug: bool = False,
             abs_unit_sec=None):
    cfg = load_yaml_config(config_file)
    cfg = complete_cfg(cfg, options, no_id=True)

    task_metadata, task_data, n_folds, unit_sec, activation, balanced = get_defs(cfg, task)

    # cancel augmentation if required
    if cancel_aug:
        cfg.freq_mask = None
        cfg.time_mask = None
        cfg.mixup = 0.0
        cfg.rotate_wav = False
    # unit_sec can be configured at runtime
    if abs_unit_sec is not None:
        unit_sec = abs_unit_sec

    # update some parameters.
    update_options = f'+task_metadata={task_metadata},+task_data={task_data}'
    update_options += f',+unit_samples={int(cfg.sample_rate * unit_sec)}'
    cfg = complete_cfg(cfg, update_options)
    # overwrite by extra command line
    options = []
    for k, v in extras.items():
        if v is not None:
            options.append(f'{k}={v}')
    options = ','.join(options)
    if len(options) > 0:
        cfg = complete_cfg(cfg, options)
    return cfg, n_folds, activation, balanced


def short_model_desc(model, head_len=5, tail_len=1):
    text = repr(model).split('\n')
    text = text[:head_len] + ['  :'] + (text[-tail_len:] if tail_len > 0 else [''])
    return '\n'.join(text)


def lineareval_downstream(config_file: str,
                          task: str,
                          options: str = '',
                          seed: int = 42,
                          lr: float | None = None,
                          hidden=(), standard_scaler=True, mixup=False,
                          epochs=None, early_stop_epochs=None,
                          unit_sec=None,
                          step='1pass',
                          loglevel: str = "info"):
    logging.basicConfig(level=getattr(logging, loglevel.upper()))

    task = task.split('-')
    if len(task) == 1:
        task, label = task[0], "label"
    elif len(task) == 2:
        task, label = task[0], task[1]
    else:
        raise ValueError(f'task {task} is invalid')

    cfg, n_folds, _, _ = make_cfg(config_file, task, options, extras={}, abs_unit_sec=unit_sec)
    lr = lr or cfg.lr_lineareval
    epochs = epochs or 200
    early_stop_epochs = early_stop_epochs or cfg.early_stop_epochs
    cfg.runtime_cfg = kwarg_cfg(lr=lr, seed=seed, hidden=hidden, standard_scaler=standard_scaler, mixup=mixup,
                                epochs=epochs, early_stop_epochs=early_stop_epochs)
    two_pass = (step != '1pass')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(seed)
    logpath = app_setup_logger(cfg, level=logging.DEBUG)  # Add this when debugging deeper: level=logging.DEBUG

    scores, ar, emb_ar, df = [], None, None, None
    for fold in range(1, n_folds + 1):
        # Dataloaders
        train_loader, valid_loader, test_loader, multi_label = create_dataloader(cfg,
                                                                                 fold=fold,
                                                                                 seed=seed,
                                                                                 label_col=label,
                                                                                 balanced_random=False,
                                                                                 pin_memory=False,
                                                                                 num_workers=4)
        dataset = train_loader(batch_size=cfg.batch_size).dataset
        # logging.info(f'Train:{len(train_loader.dataset)}, valid:{len(valid_loader.dataset)}, test:{len(test_loader.dataset)}, multi label:{multi_label}')
        classes = dataset.classes

        # Make audio representation model.
        if ar is None and step != '2pass_2_train_test':
            cache_info = get_cache_info("train", cfg.id, fold)
            if not cache_info[0].exists():
                ar = eval('evar.' + cfg.audio_repr)(cfg).to(device)

                # safe computation
                bs = cfg.batch_size
                while True:
                    try:
                        ar.precompute(device, train_loader(batch_size=bs))
                    except torch.cuda.OutOfMemoryError as e:
                        if bs == 1:
                            raise e

                        bs //= 2
                        log.debug(f"Caught OOM error, retrying with batch size={bs}.")
                        torch.cuda.empty_cache()
                        continue

                    break

                emb_ar = torch.nn.DataParallel(ar).to(device)  # TODO: why?
                log.debug(short_model_desc(ar))

        # Convert to embeddings.
        kwargs = dict(device=device, _id=cfg.id, fold=fold, label=label, cache=two_pass)
        x, y, batch_size = to_embeddings(emb_ar, train_loader, split="train", **kwargs)
        X_val, y_val, batch_size = to_embeddings(emb_ar, valid_loader, split="valid", batch_size=batch_size, **kwargs)
        X_test, y_test, batch_size = to_embeddings(emb_ar, test_loader, split="test", batch_size=batch_size, **kwargs)

        if step == '2pass_1_precompute_only':
            continue

        score, df = _one_linear_eval(x, y, X_val, y_val, X_test, y_test, batch_size=cfg.batch_size,
                                     hidden_sizes=hidden, epochs=epochs,
                                     metrics=get_metrics(label),
                                     early_stop_epochs=early_stop_epochs, lr=lr, classes=classes,
                                     standard_scaler=standard_scaler, mixup=mixup)
        scores.append(score)
        if n_folds > 1:
            log.info(f' fold={fold}: {score}')

    if step == '2pass_1_precompute_only':
        return

    mean_score = {k: np.mean([s[k] for s in scores]) for k in scores[0]}
    re_hashed = hash_text(str(cfg), L=8)
    score_file = logpath / f'{cfg.id[:-9].replace("AR_", "").replace("_", "-")}-LE_{re_hashed}.csv'
    
    if df is None:
        log.warning("Results DataFrame does not exist.")
    else:
        df.to_csv(score_file, index=False)

        if label != "tag":
            # confusion matrix
            cm = confusion_matrix(df)
            fig = px.imshow(cm, text_auto=True)
            fig.write_html(score_file.with_suffix(".html"))


    if label != "label":
        task = task + '-' + label

    report = f'Linear evaluation: {cfg.id[:-8]+re_hashed} {task} -> {mean_score}\n{cfg}\n{score_file}'

    # in order to handle different runtime configurations, we isolate the runtime config from the rest of the report
    runtime_cfg = cfg.runtime_cfg
    seed = runtime_cfg.pop("seed")
    runtime_str = json.dumps(runtime_cfg, sort_keys=True)
    runtime_id = hash_text(runtime_str, L=8)

    results = [{
        'representation': cfg.id.split('_')[-2],  # AR name
        'task': task + '-' + metric,
        'score': score,
        'seed': seed,
        'run_id': re_hashed,
        'runtime_id': runtime_id,
        'runtime_cfg': runtime_str,
        'report': report,
    } for metric, score in mean_score.items()]
    append_to_csv(RESULT_DIR / "scores.csv", results)
    log.info(report)
    log.info(f' -> {RESULT_DIR}/scores.csv')


if __name__ == '__main__':
    setup_dir([RESULT_DIR, LOG_DIR])
    fire.Fire(lineareval_downstream)
