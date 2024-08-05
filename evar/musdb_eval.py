import argparse
import json
import logging
from math import ceil
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Tuple

from tqdm import trange

import torch
import torch.nn.functional as F

import hydra
import rootutils
from lightning import LightningModule
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.data.mix import MixStemsDatamodule
from src.utils.copy import copy_to_compute_node

log = logging.getLogger(__name__)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("ckpt_path", type=Path, help="Checkpoint path")
    parser.add_argument("-o", "--output_file", type=Path, help="Output file")
    parser.add_argument("--musdb_path", type=Path,
                        default="work/16k/musdb",
                        help="Musdb path")

    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=3840)

    parser.add_argument("-n", "--normalize", action='store_true')

    parser.add_argument("-g", "--gpu", type=int, default=0)

    return parser.parse_args()


def compute_recall(similarities: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
    r"""Compute recall at all k in parallel

    Args:
        similarities (torch.Tensor): matrix of cosine similarities

    Returns:
        torch.Tensor: a tensor `recall` so that `recall[k]` = R@k, shape (num_embeddings+1,)
    """
    num_src_embeddings, num_tgt_embeddings = similarities.size()
    assert num_src_embeddings == num_tgt_embeddings, "We support only 1/1 mapping now"

    device = similarities.device

    true_indices = torch.arange(num_src_embeddings, device=device).unsqueeze(1)

    # compute the rank of the positive pair (0 => nearest neighbor, 1 => second to nearest, etc.)
    sorted_indices = similarities.argsort(dim=1)

    ranks = (sorted_indices == true_indices).long().argmax(dim=1) + 1

    recalls = torch.zeros(num_tgt_embeddings+1, dtype=torch.long, device=device)
    values, counts = torch.unique(ranks, return_counts=True)
    recalls[values] = counts
    return recalls.cumsum(dim=0).float().div_(num_src_embeddings), ranks


def compute_auc(values: torch.Tensor) -> torch.Tensor:
    r"""Compute Area Under Curve by integration using trapezoid rule.
    x-coordinates of the curve are supposed to be linspace(0, 1, len(values))

    Args:
        values (torch.Tensor): y-coordinates of the curve

    Returns:
        Area Under Curve, scalar tensor

    """
    return torch.mean(values[:-1] + values[1:]) / 2


@torch.no_grad()
def evaluate(cfg: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    num_stems = 4

    # load model
    xp_dir = cfg.ckpt_path.parents[1]
    train_cfg = OmegaConf.load(xp_dir / "config.yaml")

    log.info(f"Instantiating model <{train_cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(train_cfg.model)

    # filter state dict because it is trained as compiled module
    ckpt = torch.load(cfg.ckpt_path, map_location=torch.device("cpu"))
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)

    model.eval()

    # copy MUSDB data to compute node if it exists
    data_path = Path(__file__).parent / cfg.musdb_path
    data_path = copy_to_compute_node(data_path)

    # load data module
    data_cfg = train_cfg.data
    data_cfg["dataset_kwargs"] = dict(
        data_path=data_path,
        duration=-1,
        sample_rate=16000
    )
    dm: MixStemsDatamodule = hydra.utils.instantiate(data_cfg)
    dm.setup(stage='test')
    # pass to appropriate device
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() and cfg.gpu >= 0 else "cpu")

    model = model.to(device)
    dataset = dm.dataset
    transform = dm.transform.to(device)

    target_indices = torch.arange(num_stems, device=device).unsqueeze(1)

    all_latents = torch.zeros(num_stems * len(dataset), cfg.embed_dim, device=device)
    all_targets = torch.zeros(num_stems * len(dataset), cfg.embed_dim, device=device)
    all_predictions = torch.zeros(num_stems * len(dataset), cfg.embed_dim, device=device)

    for idx in trange(len(dataset), desc="Computing MUSDB embeddings..."):
        stems = dataset.get_audio(idx).to(device)

        # get all mixes
        mixes = stems.unsqueeze(0).expand(num_stems, num_stems, -1)[~torch.eye(num_stems, dtype=torch.bool, device=device)]
        mixes = mixes.view(num_stems, num_stems - 1, -1).sum(dim=1)

        x = torch.cat((stems, mixes), dim=0)

        x = transform(x)

        # pad x to get integer chunks
        pad_length = cfg.patch_size * ceil(x.size(-1) / cfg.patch_size)
        x = F.pad(x, (0, pad_length - x.size(-1)))

        latents = torch.zeros(num_stems, cfg.embed_dim, device=device)
        targets = torch.zeros(num_stems, cfg.embed_dim, device=device)
        predictions = torch.zeros(num_stems, cfg.embed_dim, device=device)

        total_patches = 0

        for chunk in x.split(model.encoder.img_size[-1], dim=-1):
            y_ctx = model.encoder(chunk)
            latents += y_ctx[:num_stems].sum(dim=-2).view(num_stems, cfg.embed_dim)

            y_prd = model.predictor(
                y_ctx[num_stems:],
                target_indices if target_indices is not None else conditioning
            )
            
            if torch.is_tensor(y_prd):
                y_prd = y_prd[..., :y_ctx.size(-1)]
            else:
                pred, score = y_prd  # (b, f, t, k, d), (b, f, t, k)
                head_idx = score.argmax(dim=-1, keepdim=True)
                head_idx = head_idx.expand_as(pred[..., 0, :]).unsqueeze(-2)
                y_prd = pred.gather(-2, head_idx).squeeze(-2)

            predictions += y_prd.sum(dim=-2).view(num_stems, cfg.embed_dim)

            y_tgt = model.target_encoder(chunk)
            targets += y_tgt[:num_stems].sum(dim=-2).view(num_stems, cfg.embed_dim)

            total_patches += y_tgt.size(-2)

        all_latents[num_stems * idx: num_stems * (idx + 1)] = latents / total_patches
        all_targets[num_stems * idx: num_stems * (idx + 1)] = targets / total_patches
        all_predictions[num_stems * idx: num_stems * (idx + 1)] = predictions / total_patches

    # normalize embeddings
    all_latents = F.layer_norm(all_latents, (cfg.embed_dim,))
    all_targets = F.layer_norm(all_targets, (cfg.embed_dim,))
    all_predictions = F.layer_norm(all_predictions, (cfg.embed_dim,))

    if cfg.normalize:
        all_latents = F.normalize(all_latents, p=2, dim=-1)
        all_targets = F.normalize(all_targets, p=2, dim=-1)
        all_predictions = F.normalize(all_predictions, p=2, dim=-1)
        print(all_latents.shape)

    embeddings = dict(
        context=all_latents,
        targets=all_targets
    )

    metrics = {}

    for name, latents in embeddings.items():
        distance_matrix = torch.cdist(all_predictions, latents)

        recalls, ranks = compute_recall(distance_matrix)

        for k in [1, 5, 10]:
            metrics[f"retrieval/{name}/recall@{k}"] = recalls[k]

        metrics[f"retrieval/{name}/AUC"] = compute_auc(recalls)

        # Mean/Median Normalized Rank
        norm_ranks = ranks.float() / distance_matrix.size(1)
        metrics[f"retrieval/{name}/mean rank"] = norm_ranks.mean()
        metrics[f"retrieval/{name}/median rank"] = norm_ranks.median()

    pprint(metrics)
    if cfg.output_file is None:
        return

    with cfg.output_file.open('w') as f:
        json.dump({k: v.cpu().item() for k, v in metrics.items()}, f)


def main(cfg: argparse.Namespace) -> None:
    """Main entry point for evaluation.
    """
    evaluate(cfg)


if __name__ == "__main__":
    args = parse()

    main(args)
