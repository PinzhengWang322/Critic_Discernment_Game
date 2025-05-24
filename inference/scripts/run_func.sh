#!/bin/bash

source /online1/public/support/amd/Ananconda3/2023.3/bin/activate
source /online1/public/support/amd/Ananconda3/2023.3/bin/deactivate
source /online1/public/support/amd/Ananconda3/2023.3/bin/deactivate
source /online1/public/support/amd/Ananconda3/2023.3/bin/activate vllm_2play

# 默认参数
NAME="debug"

# 检查必需参数是否已传入
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ]; then
    echo "Usage: $0 <temp> <k> <IN_PATH> <OUT_PATH> <MODEL>"
    exit 1
fi

# 解析输入参数
temp=$1
k=$2
IN_PATH=$3
OUT_PATH=$4
MODEL=$5

# 创建输出目录
DIR=$(dirname "$OUT_PATH")
mkdir -p "$DIR"

# 执行推理脚本
bash vllm_infer/data_gen.sh \
    "$IN_PATH" \
    "$OUT_PATH" \
    "$MODEL" \
    "$temp" "$k" 2>&1 | tee logs/infer3.log

# 获取当前时间（小时:分钟）并打印
date=$(date +"%H:%M")
echo "Execution completed at $date"
