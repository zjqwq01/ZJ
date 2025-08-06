#!/bin/bash

python3 sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_soc_config.yaml \
    --task_config=configs/motion_deblur_soc_config.yaml \
    --solve_type nonlinear