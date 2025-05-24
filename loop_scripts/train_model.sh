cd $TRAIN_DIR

base_model_name=$1
dataset=$2
output_dir=$3
gradient_accumulation_steps=$4

if [ -z "$base_model_name" ] || [ -z "$dataset" ] || [ -z "$output_dir" ] || [ -z "$gradient_accumulation_steps" ]; then
    echo "Error: Missing required arguments. Usage: $0 <base_model> <dataset> <output_dir> <gradient_accumulation_steps>"
    exit 1
fi

bash run_func.sh $base_model_name $dataset $output_dir $gradient_accumulation_steps

cd $MAIN_DIR