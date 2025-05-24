from transformers import AutoTokenizer
from datasets import load_from_disk
from datasets import load_dataset
import json
import random
import os
import argparse
import re
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type=str)
parser.add_argument('--out_path', type=str)
args = parser.parse_args()

# 输入和输出文件路径
input_file = args.in_path
output_file = args.out_path

assert input_file != output_file 
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

def get_critic(s):
    """
    提取 "The first mistake can be found in:" 后的句子内容。
    """
    # 正则表达式模式，匹配 "The first mistake can be found in:" 后的原句
    if s.count("The first mistake can be found in:") != 1: return None
    critic = "The first mistake can be found in:" + s.split("The first mistake can be found in:")[-1]
    return critic

def format_0shot(data):
    tmp_chat_prompts, critics = [], []
    seen_critics = set()
    for response in data['responses']:
        critic = get_critic(response)
        if (critic is None) or (len(critic) < 10): continue
        if critic in seen_critics: continue
        seen_critics.add(critic)
        chat_prompt = tokenizer.apply_chat_template(\
        [{"role": "system", "content":"Please reason step by step, and put your final answer within \\boxed{}."},
         {"role": "user", "content": data['metadata']['problem']},
        {"role": "assistant", "content": data['round1_response']},
        {"role": "user", "content": 
"""\
Please check with this critic.
"{CRITIC}"
Evaluate whether this critic is valid. Keep in mind that this critic might be misleading or irrelevant.
If you find the critic incorrect, conclude with: "\\boxed{This critic is not critical.}"
If you determine the critic to be valid, revise starting from the incorrect step, and present your revised answer within \\boxed{}.
""".replace("{CRITIC}", critic)},
        {"role": "assistant", "content": ""},], tokenize=False)

        if chat_prompt.startswith(BOS_TOKEN):
            chat_prompt = chat_prompt[len(BOS_TOKEN):]
        if chat_prompt.endswith(EOS_TOKEN):
            chat_prompt = chat_prompt[:-len(EOS_TOKEN)]
        tmp_chat_prompts.append(chat_prompt)
        critics.append(critic)
    return tmp_chat_prompts, critics

final_datas = []
for data in [json.loads(i) for i in open(input_file, 'r').readlines()]:
    chat_prompts, critics = format_0shot(data)
    del data['responses']
    for chat_prompt, critic in zip(chat_prompts, critics):
        data["prompt"] = chat_prompt
        data["critic"] = critic
        final_datas.append(copy.deepcopy(data))
    

# 打开输入文件并逐行处理
with open(output_file, 'w') as outfile:
    cnt = 0
    for data in final_datas:
        outfile.write(json.dumps(data) + '\n')
        cnt += 1
        if cnt % 10000 ==0: print(f"processing {cnt} lines")

print("Processing complete. Modified data saved to", output_file)
