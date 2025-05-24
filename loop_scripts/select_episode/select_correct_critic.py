import argparse
import json
import re
from transformers import AutoTokenizer
import multiprocessing as mp
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument('--critic_gen_path', type=str, required=True)
parser.add_argument('--good_prover_path', type=str, required=True)
parser.add_argument('--out_path', type=str, required=True)
parser.add_argument('--success_ratio', type=float, default=0.5)
parser.add_argument('--prover_gen_num', type=float, default=4)
parser.add_argument('--type', type=str, choices=["correct", "hack"], required=True)
args = parser.parse_args()

def get_critic(s):

    if s.count("The first mistake can be found in:") != 1: return None
    critic = "The first mistake can be found in:" + s.split("The first mistake can be found in:")[-1]
    return critic

tokenizer_path = os.environ['BASE_MODEL_NAME']
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
            critic_step = get_critic(response)
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
        critric_step_dict[critc_hash_id]['right_cnt'] +=1

    sft_datas = []
    idx = 0
    for _, value in critric_step_dict.items():
        idx += 1
        if args.type == "correct":
            if value['right_cnt'] >= len(value['responses']) * args.prover_gen_num * args.success_ratio:
                prompt = value["prompt"].split("<|start_header_id|>user<|end_header_id|>\n\n")[1].split("<|eot_id|>")[0].strip()
                for response in value['responses']:
                    sft_datas.append(dict(
                                        instruction = prompt,
                                        input = "",
                                        output = response,
                                        system = "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n",
                                    ))
        else:
            if value['right_cnt'] <= len(value['responses']) * args.prover_gen_num * (1 - args.success_ratio):
                prompt = value["prompt"].split("<|start_header_id|>user<|end_header_id|>\n\n")[1].split("<|eot_id|>")[0].strip()
                for response in value['responses']:
                    sft_datas.append(dict(
                                        instruction = prompt,
                                        input = "",
                                        output = response,
                                        system = "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n",
                                    ))
        
    return sft_datas

if __name__ == '__main__':
    critic_datas = [json.loads(line) for line in open(args.critic_gen_path).readlines()]
    rigth_prover_datas = [json.loads(line) for line in open(args.good_prover_path).readlines()]

    critric_step_dict = record_critic_data(critic_datas)
    sft_datas =  select_data(rigth_prover_datas, critric_step_dict)
    
    with open(args.out_path, 'w') as f:
        f.write(json.dumps(sft_datas, indent=4))


