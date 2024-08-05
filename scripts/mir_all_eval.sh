#!/bin/bash
CONFIG_FILE=config/stems.yaml

# checkpoint to evaluate (made absolute)
WEIGHT_FILE="$(cd "$(dirname "$1")" || exit 1; pwd)/$(basename "$1")"

# temporary file to write the results into (useful if the script is called from a subprocess)
TMP_FILE=$2

# device
if [ -n "$3" ]; then
  GPU="$3"
else
  GPU=0
fi

cd evar || exit 1

CUDA_VISIBLE_DEVICES=$GPU python 2pass_lineareval.py $CONFIG_FILE giantsteps_stems-key batch_size=64,weight_file="$WEIGHT_FILE" "${@:4}"
CUDA_VISIBLE_DEVICES=$GPU python 2pass_lineareval.py $CONFIG_FILE gtzan_stems-genre batch_size=64,weight_file="$WEIGHT_FILE" "${@:4}"
CUDA_VISIBLE_DEVICES=$GPU python 2pass_lineareval.py $CONFIG_FILE mtt-tag batch_size=64,weight_file="$WEIGHT_FILE" "${@:4}"
CUDA_VISIBLE_DEVICES=$GPU python 2pass_lineareval.py $CONFIG_FILE nsynth batch_size=64,weight_file="$WEIGHT_FILE" "${@:4}"

python summarize.py "$WEIGHT_FILE" "$TMP_FILE"

# retrieval evaluation on MUSDB
CUDA_VISIBLE_DEVICES=$GPU python musdb_eval.py -n "$WEIGHT_FILE" -o "$TMP_FILE.json"
