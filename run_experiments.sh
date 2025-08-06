#!/bin/bash

#!/bin/bash

# echo "开始运行实验..."

# 使用反斜杠 \ 将长命令换行，以提高可读性
python3 sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_soc_config.yaml \
    --task_config=configs/phase_retrieval_soc_config.yaml \
    --solve_type nonlinear

# echo "实验运行结束。"

# # 当任何命令失败时，立即退出脚本
# set -e

# # --- 公共参数定义 ---
# # 将命令中不变的部分定义为变量，方便修改
# BASE_CMD="python3 sample_condition.py"
# MODEL_CONFIG="--model_config=configs/model_config.yaml"
# DIFFUSION_CONFIG="--diffusion_config=configs/diffusion_soc_config.yaml"

# # --- 任务列表定义 ---
# # 将所有需要运行的任务配置文件放入一个数组
# TASKS=(
#     "configs/super_resolution_soc_config.yaml"
#     "configs/colorization_soc_config.yaml"
#     "configs/inpainting_soc_config.yaml"
#     "configs/super_resolution_inpainting_soc_config.yaml"
# )

# # --- 主循环 ---
# # 遍历任务列表，为每个任务执行所有参数组合
# echo "启动批量实验..."

# for task_config in "${TASKS[@]}"; do
#     # 从文件名中提取任务名称，用于打印日志
#     task_name=$(basename "$task_config" .yaml)

#     echo ""
#     echo "=================================================================="
#     echo "=== 开始执行任务: $task_name"
#     echo "=================================================================="
#     echo ""

#     # --- 运行组合 1: linear ---
#     echo "--> 组合 1/5: --solve_type linear"
#     $BASE_CMD $MODEL_CONFIG $DIFFUSION_CONFIG --task_config="$task_config" --solve_type="linear"
#     echo "--> 组合 1/5 完成."
#     echo ""

#     # --- 运行组合 2: nonlinear ---
#     echo "--> 组合 2/5: --solve_type nonlinear"
#     $BASE_CMD $MODEL_CONFIG $DIFFUSION_CONFIG --task_config="$task_config" --solve_type="nonlinear"
#     echo "--> 组合 2/5 完成."
#     echo ""

#     # --- 运行组合 3: nonlinear_finitegamma, gamma=1e3 ---
#     echo "--> 组合 3/5: --solve_type nonlinear_finitegamma --gamma 1e3"
#     $BASE_CMD $MODEL_CONFIG $DIFFUSION_CONFIG --task_config="$task_config" --solve_type="nonlinear_finitegamma" --gamma=1e3
#     echo "--> 组合 3/5 完成."
#     echo ""
    
#     # --- 运行组合 4: nonlinear_finitegamma, gamma=1e5 ---
#     echo "--> 组合 4/5: --solve_type nonlinear_finitegamma --gamma 1e5"
#     $BASE_CMD $MODEL_CONFIG $DIFFUSION_CONFIG --task_config="$task_config" --solve_type="nonlinear_finitegamma" --gamma=1e5
#     echo "--> 组合 4/5 完成."
#     echo ""

#     # --- 运行组合 5: nonlinear_finitegamma, gamma=1e7 ---
#     echo "--> 组合 5/5: --solve_type nonlinear_finitegamma --gamma 1e7"
#     $BASE_CMD $MODEL_CONFIG $DIFFUSION_CONFIG --task_config="$task_config" --solve_type="nonlinear_finitegamma" --gamma=1e7
#     echo "--> 组合 5/5 完成."
#     echo ""

#     echo "=== 任务 $task_name 执行完毕。"
# done

# echo "=================================================================="
# echo "=== 所有任务已成功执行完毕！"
# echo "=================================================================="