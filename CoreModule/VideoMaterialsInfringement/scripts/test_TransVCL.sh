#!/usr/bin/env bash
DATASET=VCSL
MODEL=model
FEATDIR=data/VCSL/featuress/

python run.py \
       --model-file transvcl/weights/${MODEL}.pth \
       --feat-dir data/${DATASET}/featuress/eff256d/ \
       --test-file data/${DATASET}/pair_file.csv \
       --save-file results/${MODEL}/${DATASET}/result.json

       