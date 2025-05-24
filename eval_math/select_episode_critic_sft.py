import argparse
import json
import re
from transformers import AutoTokenizer
import multiprocessing as mp
from tqdm import tqdm

# 初始化参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--critic_gen_path', type=str, required=True)
parser.add_argument('--good_prover_path', type=str, required=True)
parser.add_argument('--out_path', type=str, required=True)
parser.add_argument('--success_ratio', type=float, default=0.5)
parser.add_argument('--prover_gen_num', type=float, default=8)
parser.add_argument('--type', type=str, choices=["correct", "hack"], required=True)
args = parser.parse_args()

def get_critic_step(s):
    pattern = r'The first mistake can be found in:\s*"(.*?)"'  # 捕获引号内的内容
    match = re.search(pattern, s, re.DOTALL)  # 启用 DOTALL 匹配多行内容
    
    if match:
        # 返回提取的原句
        return match.group(1).strip()
    else:
        return None

tokenizer_path = "/online1/ycsc_lijt1/lijt1/wpz/hf_models/Qwen2.5-Math-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
if "Meta-Llama-3-8B" in tokenizer_path:
    EOS_TOKEN = "<|eot_id|>"
elif "Qwen" in tokenizer_path:
    EOS_TOKEN = "<|im_end|>\n"
else:
    EOS_TOKEN = tokenizer.eos_token

if "Qwen" in tokenizer_path:
    BOS_TOKEN = "<|im_start|>"
else:
    BOS_TOKEN = tokenizer.bos_token


def record_critic_data(datas):
    critric_step_dict = dict()
    for data in datas:
        responses = data['responses']
        for response in responses:
            critic_prompt = data["prompt"]
            round1_response = data["round1_response"]
            critic_step = get_critic_step(response)
            if critic_step is None: continue
            critc_hash_id = hash(round1_response + critic_step)
            if critc_hash_id not in critric_step_dict:
                critric_step_dict[critc_hash_id] = dict(
                    responses = [response],
                    prompt = critic_prompt,
                    right_cnt = 0
                )
            else:
                critric_step_dict[critc_hash_id]['responses'].append(response)
    return critric_step_dict

def select_data(good_prover_datas, critric_step_dict):
    for prover_data in good_prover_datas:
        critic_step = prover_data["critic"]
        round1_response = prover_data["round1_response"]
        if critic_step is None: continue
        critc_hash_id = hash(round1_response + critic_step)
        if prover_data["metadata"]["gold_num_answer"] not in ["A", "B", "C", "D"]:
            critric_step_dict[critc_hash_id]['right_cnt'] +=1

    sft_datas = []
    idx = 0
    for _, value in critric_step_dict.items():
        idx += 1
        if args.type == "correct":
            if value['right_cnt'] >= len(value['responses']) * args.prover_gen_num * args.success_ratio:
                for response in value['responses']:
                    sft_datas.append({"prompt": BOS_TOKEN + value["prompt"], 
                                    "completion": response + EOS_TOKEN, "idx": idx})
        else:
            if value['right_cnt'] <= len(value['responses']) * args.prover_gen_num * (1 - args.success_ratio):
                for response in value['responses']:
                    sft_datas.append({"prompt": BOS_TOKEN + value["prompt"], 
                                    "completion": response + EOS_TOKEN, "idx": idx})
        
    return sft_datas

if __name__ == '__main__':
    critic_datas = [json.loads(line) for line in open(args.critic_gen_path).readlines()]
    rigth_prover_datas = [json.loads(line) for line in open(args.good_prover_path).readlines()]

    critric_step_dict = record_critic_data(critic_datas)
    sft_datas =  select_data(rigth_prover_datas, critric_step_dict)
    
    with open(args.out_path, 'w') as f:
        for entry in sft_datas:
            f.write(json.dumps(entry) + '\n')


"""
python select_episode_critic_sft.py \
--critic_gen_path /online1/ycsc_lijt1/lijt1/wpz/Critic_CDG/dataset/NuminaMath/critic/output/train0/correct_v3_8k_noref.jsonl \
--good_prover_path /online1/ycsc_lijt1/lijt1/wpz/Critic_CDG/dataset/NuminaMath/prover_round2/outputs/temp0.95_8/selected/train0/correct_v3.1_cleaned_0.85_3.jsonl \
--success_ratio 0.1 \
--out_path /online1/ycsc_lijt1/lijt1/wpz/Critic_CDG/SFT/data/critic/correct_v3.1_cleaned_0.85_3-ration0.1.jsonl \
--type correct
"""