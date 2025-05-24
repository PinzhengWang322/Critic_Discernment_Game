export GPU_NUM=8
temp=0
k=1


INPUT_DIR_LIST=(
    "../data/test-data"
    "../data/test-data"
)

OUT_DIR_LIST=(
    "../data/test-data-out"
    "../data/test-data-out"
)

MODEL_LIST=(
    "/path/to/Llama3.1-8b-Instruct"
    "/path/to/CDG"
)

NAME_LIST=(
    "instrcut"
    "CDG"
)

if [ ${#INPUT_DIR_LIST[@]} -ne ${#OUT_DIR_LIST[@]} ] || [ ${#INPUT_DIR_LIST[@]} -ne ${#MODEL_LIST[@]} ]; then
    echo "Error: INPUT_DIR_LIST, OUT_DIR_LIST and MODEL_LIST must have the same length."
    exit 1
fi

ulimit -u 20000

for idx in "${!INPUT_DIR_LIST[@]}"; do
    INPUT_DIR=${INPUT_DIR_LIST[idx]}
    OUT_DIR=${OUT_DIR_LIST[idx]}
    MODEL=${MODEL_LIST[idx]}
    MODEL_NAME=${NAME_LIST[idx]}
    OUTPUT_DIR_BASE="$OUT_DIR/$MODEL_NAME/data"
    RESULT_FILE="$OUT_DIR/$MODEL_NAME/result.log"

    mkdir -p "$OUTPUT_DIR_BASE"


    total_score=0
    count=0
    file_counter=0

    for json_file in "$INPUT_DIR"/*.jsonl; do

        dataset_name=$(basename "$json_file" .jsonl)

        OUTPUT_PATH="$OUTPUT_DIR_BASE/${dataset_name}.jsonl"
        mkdir -p $OUTPUT_DIR_BASE

        bash vllm_infer/data_gen_test.sh \
        "$json_file" \
        "$OUTPUT_PATH" \
        "$MODEL" \
        $temp $k 2>&1 | tee logs/infer2.log

        score=$(python eval_math/eval.py --path "$OUTPUT_PATH" | tee logs/eval_temp.log | grep -oP 'strict-match score: \K[0-9.]+')
        
        if [ -n "$score" ]; then
            echo "$dataset_name score: $score" >> "$RESULT_FILE"
            total_score=$(echo "$total_score + $score" | bc)
            count=$((count + 1))
        else
            echo "Failed to extract score for $dataset_name" >> "$RESULT_FILE"
        fi
    done

    if [ $count -gt 0 ]; then
        avg_score=$(echo "scale=4; $total_score / $count" | bc)
        echo "Average score for $MODEL_NAME: $avg_score" >> "$RESULT_FILE"
    else
        echo "No valid scores found for $MODEL_NAME" >> "$RESULT_FILE"
    fi
done


