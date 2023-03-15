#!/bin/bash
export OMP_NUM_THREADS=1

set -ue

name=l2
expdir=exp
python run_pretrain.py \
    --upstream alhubert \
    --upstream_config "config/alhubert/config_model_l2.yaml" \
    --config "config/alhubert/config_runner.yaml" \
    --expname $name \
    --expdir $expdir/$name
