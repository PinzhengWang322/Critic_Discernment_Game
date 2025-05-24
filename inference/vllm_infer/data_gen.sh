in_path=$1
out_path=$2
model_path=$3

temp=$4
K=$5

python vllm_infer/vllm_infer.py \
--jsonl_path $in_path \
--k $K \
--temperature $temp \
--num_gpus $GPU_NUM \
--model_path $model_path \
--out_path $out_path 