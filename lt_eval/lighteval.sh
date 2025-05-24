#!/bin/bash

# source /online1/ycsc_lijt1/lijt1/wpz/GRPO/open-r1-main/openr1/bin/activate
# 所有模型及对应简称
declare -A MODELS=(
    ["llama-CDG2"]="/online1/ycsc_lijt1/lijt1/wpz/Llama_CDG/data_final/V1.3.0/ablation_ckpt/CDG_sl_bsz256"
)

# 温度列表
TEMPERATURES=(0 0.75)

# 任务列表
TASKS=(
    "custom|gsm8k|0|0"
    "custom|math_500|0|0"
)

# Custom task脚本路径
CUSTOM_TASK_SCRIPT="openr1/evaluate.py"

# CPU 设置
NUM_CPUS=8

# 输出和日志目录
BASE_OUTPUT_DIR="./results"
BASE_LOG_DIR="./logs"

# 创建日志和输出目录
mkdir -p "$BASE_LOG_DIR"

# 开始循环
for model_name in "${!MODELS[@]}"; do
    model_path="${MODELS[$model_name]}"
    for temp in "${TEMPERATURES[@]}"; do
        for task in "${TASKS[@]}"; do

            task_clean=$(echo "$task" | tr '|:/' '__')  
            output_dir="$BASE_OUTPUT_DIR/$model_name/temp${temp}/${task_clean}"
            log_path="$BASE_LOG_DIR/${model_name}_temp${temp}_${task_clean}.log"

            ray stop --force
            ray start --head --num-cpus=$NUM_CPUS


            MODEL_ARGS="pretrained=$model_path,dtype=bfloat16,max_model_length=8192,data_parallel_size=4,gpu_memory_utilization=0.85,generation_parameters={max_new_tokens:8192,temperature:$temp}"
            lighteval vllm $MODEL_ARGS "$task" \
                --custom-tasks $CUSTOM_TASK_SCRIPT \
                --use-chat-template \
                --save-details \
                --output-dir "$output_dir" 2>&1 | tee "$log_path"
            
        done
    done
done

# 最后停止 Ray
ray stop --force
