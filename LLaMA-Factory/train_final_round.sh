export TRAIN_DATA_DIR=/path/to/train_data
export TRAIN_EPOCH=1
export LR=1e-06

base_model_name=/path/to/Meta-Llama-3.1-8B-Instruct
dataset="EI_round1-loop1,correct-round2-loop1,mislead-round2-loop1,EI_round1-loop2,correct-round2-loop2,mislead-round2-loop2"
output_dir=/path/to/output_dir
gradient_accumulation_steps=16

if [ -z "$base_model_name" ] || [ -z "$dataset" ] || [ -z "$output_dir" ] || [ -z "$gradient_accumulation_steps" ]; then
    echo "Error: Missing required arguments. Usage: $0 <base_model> <dataset> <output_dir> <gradient_accumulation_steps>"
    exit 1
fi

bash run_func.sh $base_model_name $dataset $output_dir $gradient_accumulation_steps

cd $MAIN_DIR