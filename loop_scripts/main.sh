#!/bin/bash

module load amd/cuda/12.2 
export GPU_NUM=8
export MAIN_DIR="/path/to/Critic_Discernment_Game"
export INFERENCE_DIR="$MAIN_DIR/inference"
export TRAIN_DIR="$MAIN_DIR/LLaMA-Factory"
export BASE_MODEL_NAME=/path/to/Meta-Llama-3.1-8B-Instruct

MATHQA_PATH="$MAIN_DIR/data/train.jsonl"
BASE_MODEL=$BASE_MODEL_NAME
PROVER_MODEL=$BASE_MODEL; CORRECT_CRITIC_MODEL=$BASE_MODEL; MISLEAD_CRITIC_MODEL=$BASE_MODEL

exp_name=CDG_ReST
start_loop=1
end_loop=3

SAVE_DIR=$MAIN_DIR/$exp_name
DATA_DIR=$SAVE_DIR/data
CKP_DIR=$SAVE_DIR/ckpt
export TRAIN_DATA_DIR=$SAVE_DIR/train_data
mkdir -p $TRAIN_DATA_DIR

PROVER_DATASET=""; CORRECT_DATASET=""; MISLEAD_DATASET="";
LOG_FILE=logs/time_${exp_name}.log
mkdir -p logs

for (( loop=$start_loop; loop<=$end_loop; loop++ ))
do
    TMP_DATA_DIR=$DATA_DIR/loop$loop
    TMP_CKP_DIR=$CKP_DIR/loop$loop
    mkdir -p $TMP_DATA_DIR/outputs; mkdir -p $TMP_DATA_DIR/prompts; mkdir -p $TMP_DATA_DIR/selected
    mkdir -p $TMP_CKP_DIR


    ########################################
    # STEP1: prover round1 generate and split
    start_time=$(date +%s)
    echo "[LOOP $loop] STEP1 START: $(date)" >> $LOG_FILE

    bash loop_scripts_local/gen_data.sh 0.95 1 $MATHQA_PATH $TMP_DATA_DIR/outputs/round1.jsonl $PROVER_MODEL

    python eval_math/split.py --path $TMP_DATA_DIR/outputs/round1.jsonl
    RIGHT_PATH=$TMP_DATA_DIR/outputs/right_round1.jsonl
    WRONG_PATH=$TMP_DATA_DIR/outputs/wrong_round1.jsonl

    end_time=$(date +%s)
    echo "[LOOP $loop] STEP1 END: $(date) | Duration: $((end_time - start_time)) seconds" >> $LOG_FILE


    #########################################
    # STEP2: generate critics
    start_time=$(date +%s)
    echo "[LOOP $loop] STEP2 START: $(date)" >> $LOG_FILE

    python loop_scripts_local/data_process/make_critic.py \
    --in_path $TMP_DATA_DIR/outputs/right_round1.jsonl \
    --out_path $TMP_DATA_DIR/prompts/mislead-critic.jsonl

    python loop_scripts_local/data_process/make_critic.py \
    --in_path $TMP_DATA_DIR/outputs/wrong_round1.jsonl \
    --out_path $TMP_DATA_DIR/prompts/correct-critic.jsonl

    bash loop_scripts_local/gen_data.sh 0.95 4 $TMP_DATA_DIR/prompts/mislead-critic.jsonl $TMP_DATA_DIR/outputs/mislead-critic.jsonl $MISLEAD_CRITIC_MODEL
    bash loop_scripts_local/gen_data.sh 0.95 8 $TMP_DATA_DIR/prompts/correct-critic.jsonl $TMP_DATA_DIR/outputs/correct-critic.jsonl $CORRECT_CRITIC_MODEL

    end_time=$(date +%s)
    echo "[LOOP $loop] STEP2 END: $(date) | Duration: $((end_time - start_time)) seconds" >> $LOG_FILE

    #########################################
    # STEP3: generate prover round2
    start_time=$(date +%s)
    echo "[LOOP $loop] STEP3 START: $(date)" >> $LOG_FILE

    python loop_scripts_local/data_process/make_prover_round2.py \
    --in_path $TMP_DATA_DIR/outputs/mislead-critic.jsonl \
    --out_path $TMP_DATA_DIR/prompts/mislead-prover-round2.jsonl

    python loop_scripts_local/data_process/make_prover_round2.py \
    --in_path $TMP_DATA_DIR/outputs/correct-critic.jsonl \
    --out_path $TMP_DATA_DIR/prompts/correct-prover-round2.jsonl

    bash loop_scripts_local/gen_data.sh 0.95 4 $TMP_DATA_DIR/prompts/mislead-prover-round2.jsonl $TMP_DATA_DIR/outputs/mislead-prover-round2.jsonl $PROVER_MODEL
    bash loop_scripts_local/gen_data.sh 0.95 4 $TMP_DATA_DIR/prompts/correct-prover-round2.jsonl $TMP_DATA_DIR/outputs/correct-prover-round2.jsonl $PROVER_MODEL

    end_time=$(date +%s)
    echo "[LOOP $loop] STEP3 END: $(date) | Duration: $((end_time - start_time)) seconds" >> $LOG_FILE

    #########################################
    # STEP4: select episode & make train data
    start_time=$(date +%s)
    echo "[LOOP $loop] STEP4 START: $(date)" >> $LOG_FILE

    python loop_scripts_local/select_episode/make_info.py --path $TRAIN_DATA_DIR/dataset_info.json --loop $loop

    # EI prover round1
    python loop_scripts_local/select_episode/convert_round1.py \
    --in_path $TMP_DATA_DIR/outputs/right_round1.jsonl \
    --out_path $TRAIN_DATA_DIR/EI-loop${loop}.json

    # prover round2
    python loop_scripts_local/select_episode/select_round2.py \
    --in_path $TMP_DATA_DIR/outputs/mislead-prover-round2.jsonl \
    --out_path $TMP_DATA_DIR/selected/mislead-prover-round2.jsonl \
    --type mislead_resist

    python loop_scripts_local/select_episode/select_round2.py \
    --in_path $TMP_DATA_DIR/outputs/correct-prover-round2.jsonl \
    --out_path $TMP_DATA_DIR/selected/correct-prover-round2.jsonl \
    --type correct

    python loop_scripts_local/select_episode/convert_round2.py \
    --in_path $TMP_DATA_DIR/selected/mislead-prover-round2.jsonl \
    --out_path $TRAIN_DATA_DIR/mislead-prover-round2-loop${loop}.json \
    --mask_history

    python loop_scripts_local/select_episode/convert_round2.py \
    --in_path $TMP_DATA_DIR/selected/correct-prover-round2.jsonl \
    --out_path $TRAIN_DATA_DIR/correct-prover-round2-loop${loop}.json \
    --mask_history

    # critic
    python loop_scripts_local/select_episode/select_correct_critic.py \
    --critic_gen_path $TMP_DATA_DIR/outputs/correct-critic.jsonl \
    --good_prover_path $TMP_DATA_DIR/selected/correct-prover-round2.jsonl \
    --success_ratio 0.5 \
    --out_path $TRAIN_DATA_DIR/correct-critic-loop${loop}.json \
    --type correct

    python loop_scripts_local/select_episode/select_mislead_critic.py \
    --in_critic_path $TMP_DATA_DIR/outputs/mislead-critic.jsonl \
    --in_prover_path $TMP_DATA_DIR/outputs/mislead-prover-round2.jsonl \
    --out_path $TRAIN_DATA_DIR/mislead-critic-loop${loop}.json \
    --hack_ratio 0.75

    end_time=$(date +%s)
    echo "[LOOP $loop] STEP4 END: $(date) | Duration: $((end_time - start_time)) seconds" >> $LOG_FILE

    #########################################
    # STEP5: tune every agents
    start_time=$(date +%s)
    echo "[LOOP $loop] STEP5 START: $(date)" >> $LOG_FILE

    # Accumulate datasets
    if [ -n "$PROVER_DATASET" ]; then
        PROVER_DATASET+=","
    fi
    PROVER_DATASET+="EI_round1-loop${loop},correct-round2-loop${loop},mislead-round2-loop${loop}"

    if [ -n "$MISLEAD_DATASET" ]; then
        MISLEAD_DATASET+=","
    fi
    MISLEAD_DATASET+="mislead_critic-loop${loop}"

    if [ -n "$CORRECT_DATASET" ]; then
        CORRECT_DATASET+=","
    fi
    CORRECT_DATASET+="correct_critic-loop${loop}"

    if [ "$loop" -ge 2 ]; then
        gradient_accumulation_prover=1
        gradient_accumulation=1
    else
        gradient_accumulation_prover=8
        gradient_accumulation=1
    fi

    if [ "$loop" -ne 1 ]; then
        export TRAIN_EPOCH=1; export LR=5e-06
    else
        export TRAIN_EPOCH=1; export LR=1e-06
    fi
    bash loop_scripts_local/train_model.sh $BASE_MODEL "$PROVER_DATASET" $TMP_CKP_DIR/prover $gradient_accumulation_prover


    export TRAIN_EPOCH=1; export LR=5e-06
    bash loop_scripts_local/train_model.sh $BASE_MODEL "$MISLEAD_DATASET" $TMP_CKP_DIR/mislead-critic $gradient_a59
    ccumulation


    export TRAIN_EPOCH=2; export LR=5e-06
    bash loop_scripts_local/train_model.sh $BASE_MODEL "$CORRECT_DATASET" $TMP_CKP_DIR/correct-critic $gradient_accumulation
    
    PROVER_MODEL=$TMP_CKP_DIR/prover
    CORRECT_CRITIC_MODEL=$TMP_CKP_DIR/correct-critic
    MISLEAD_CRITIC_MODEL=$TMP_CKP_DIR/mislead-critic

    end_time=$(date +%s)
    echo "[LOOP $loop] STEP5 END: $(date) | Duration: $((end_time - start_time)) seconds" >> $LOG_FILE
    
done
