#!/bin/bash
set -e

work_dir=./t2i_experiments/train/pixeldit
np=8


if [[ $1 == *.yaml ]]; then
    config=$1
    shift
else
    config="configs/PixelDiT_512px_pixel_diffusion_stage1.yaml"
    echo "Only support .yaml files, but got $1. Defaulting to --config_path=$config"
fi

torchrun --nproc_per_node=$np --master_port=29501 \
        train.py \
        --config_path=$config \
        --work_dir=$work_dir \
        --name=pixeldit_run \
        --resume_from=latest \
        "$@"
