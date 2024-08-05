# Joint Embedding Predictive Architecture for Musical Stem Compatibility Estimation

This repository contains the code of the [Stem-JEPA paper](), which has been accepted at [ISMIR 2024](https://ismir2024.ismir.net/).

![model.png](https://github.com/SonyCSLParis/Stem-JEPA/blob/master/images/model.png)

## Setup

Just clone this repository and install the dependencies.
Both `requirements.txt` and `environment.yml` files are provided.

To evaluate the quality of the representations learned by our model, we rely on [EVAR](https://github.com/nttcslab/eval-audio-repr).
We did a few modifications to their original codebase to adapt it to more music-related downstream tasks.
In order to evaluate the representations, download and setup the different datasets following the README of the original repository.


## Usage

We rely on [Hydra](https://hydra.cc/) and [Dora](https://github.com/facebookresearch/dora/tree/main) to handle configurations and job scheduling.
In particular, our codebase is built on top of this great [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template).

### Data

To specify the path to your dataset, create a YAML file `my_data.yaml` in subfolder `configs/data`, following the `example.yaml` provided.
The `data_path` you indicate will be recursively explored while instantiating the PyTorch dataset.

The dataset expects audio files to be WAV files with 4 channels (bass, drums, other, vocals).
We used a sampling rate of 16 kHz in our work, but one can choose another one (waveforms will be converted to mel-spectrograms anyway).
The sampling rate can be overriden in the YAML file or through command-line directly.

**Note:** In practice, the code can handle audios with more than 4 stems.
You should then modify the number of sources for the predictor accordingly.
Just make sure that the order of the channels is always the same among all your files.
Indeed, we internally use the channel index as target instrument class to condition the predictor,
so swapping channels will lead to wrong conditioning.

### Training locally

We use `dora` to launch training. If you want to start training a model locally, just type:
```shell
dora run data=my_data logger=csv trainer=gpu
```

You can use all Hydra capabilities to customize the configuration of the model, e.g.
```shell
dora run data=my_data data.dataloader.batch_size=128 logger=wandb trainer=gpu model.predictor.num_sources=6
```
to train a model with 6 stems, a batch size of 128 and monitoring the progress of your runs using Weights & Biases.

We refer to the documentation of Hydra for more advanced usage.

### Launching jobs on a cluster

We take advantage of Dora to launch jobs seamlessly on SLURM clusters.
Adjust or override `slurm/defaul.yaml`, then use `dora launch -p <partition>` instead of `dora run`.
We refer to the documentation of Dora for more advanced usage, such as resuming experiment or grid searches.

#### Copy data to compute node

On some clusters, the bandwidth between the disk the data is stored on and the compute node can be a huge bottleneck.
In that case, it may be beneficial to copy the whole data from the login node to the compute node before starting training.
To do so, just set `data.local_dir=true` in your command.

## Cite

If you use this paper in your own work, please cite Stem-JEPA.
```
@inproceedings{StemJEPA,
address = {San Francisco},
archivePrefix = {arXiv},
arxivId = {2309.02265},
author = {Riou, Alain and Lattner, Stefan and Hadjeres, Ga{\"{e}}tan and Anslow, Michael and Peeters, Geoffroy},
booktitle = {Proceedings of the 25th International Society for Music Information Retrieval Conference},
eprint = {2309.02265},
month = {nov},
pages = {535--544},
publisher = {ISMIR},
title = {{Stem-JEPA: A Joint-Embedding Predictive Architecture for Musical Stem Compatibility Estimation}},
url = {https://doi.org/10.5281/zenodo.10265343},
year = {2024}
}
```