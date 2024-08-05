#!/bin/bash

CONFIG_FILE=config/stems.yaml

# checkpoint to evaluate (made absolute)
ORIG_WEIGHT_FILE="$(cd "$(dirname "$1")" || exit 1; pwd)/$(basename "$1")"

# temporary file to write the results into (useful if the script is called from a subprocess)
TMP_FILE=$2

# create a temporary checkpoint to ensure that the checkpoint is not overriden during evaluation
WEIGHT_FILE=$(mktemp "$ORIG_WEIGHT_FILE".XXXX)

cp "$ORIG_WEIGHT_FILE" "$WEIGHT_FILE" || exit 1

# remove temporary file when the program terminates (similar to a `finally` clause)
trap 'rm $WEIGHT_FILE' EXIT

cd evar || exit 1

CUDA_VISIBLE_DEVICES=0 python lineareval.py $CONFIG_FILE giantsteps_stems-key batch_size=16,weight_file="$WEIGHT_FILE" "${@:3}"
CUDA_VISIBLE_DEVICES=0 python lineareval.py $CONFIG_FILE gtzan_stems-genre batch_size=16,weight_file="$WEIGHT_FILE" "${@:3}" --step=2pass
CUDA_VISIBLE_DEVICES=0 python lineareval.py $CONFIG_FILE gtzan_stems-tempo batch_size=16,weight_file="$WEIGHT_FILE" "${@:3}"

python summarize.py "$WEIGHT_FILE" "$TMP_FILE"

# retrieval evaluation on MUSDB
CUDA_VISIBLE_DEVICES=0 python musdb_eval.py -n "$WEIGHT_FILE" -o "$TMP_FILE.json"
