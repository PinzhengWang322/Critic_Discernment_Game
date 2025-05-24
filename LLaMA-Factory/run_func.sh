#!/bin/bash

# 从外部参数传入 base_model, dataset, output_dir 和 gradient_accumulation_steps
base_model_name=$1
dataset=$2
output_dir=$3
gradient_accumulation_steps=$4

source /online1/public/support/amd/Ananconda3/2023.3/bin/activate
source /online1/public/support/amd/Ananconda3/2023.3/bin/deactivate
source /online1/public/support/amd/Ananconda3/2023.3/bin/deactivate
source /online1/public/support/amd/Ananconda3/2023.3/bin/activate MC

# 默认参数
template="llama3"
dataset_dir=$TRAIN_DATA_DIR
log_file="logs/$(basename $output_dir).log"

# 检查必要参数是否提供
if [ -z "$base_model_name" ] || [ -z "$dataset" ] || [ -z "$output_dir" ] || [ -z "$gradient_accumulation_steps" ]; then
    echo "Error: Missing required arguments. Usage: $0 <base_model> <dataset> <output_dir> <gradient_accumulation_steps>"
    exit 1
fi

# 打印参数信息
echo "Processing dataset: $dataset"
echo "Output directory: $output_dir"
echo "Base model: $base_model_name"
echo "Gradient Accumulation Steps: $gradient_accumulation_steps"

# 执行训练脚本
FORCE_TORCHRUN=1 llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path ${base_model_name} \
    --preprocessing_num_workers 16 \
    --finetuning_type full \
    --template ${template} \
    --dataset_dir ${dataset_dir} \
    --dataset ${dataset} \
    --cutoff_len 4096 \
    --learning_rate $LR \
    --num_train_epochs $TRAIN_EPOCH \
    --max_samples 3000000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 5000 \
    --warmup_steps 0 \
    --optim adamw_torch \
    --packing False \
    --report_to none \
    --output_dir ${output_dir} \
    --bf16 True \
    --overwrite_cache True \
    --overwrite_output_dir True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --deepspeed examples/deepspeed/ds_z2_config.json  \
    --save_only_model \
    2>&1 | tee $log_file

echo "Finished processing dataset: $dataset"
sleep 60
echo "All tasks completed."
