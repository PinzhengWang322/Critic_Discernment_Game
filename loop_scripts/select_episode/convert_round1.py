from transformers import AutoTokenizer
from datasets import load_from_disk
from datasets import load_dataset
import argparse
import json
import random
import os

def sample_lst(input_list, target_size):
    input_size = len(input_list)
    
    if input_size < target_size:
        return random.choices(input_list, k=target_size)
    elif input_size == target_size:
        return input_list
    else:
        return random.sample(input_list, target_size)

def has_repeated_substring_rolling_hash(s: str, L: int, times: int) -> bool:
    n = len(s)
    if L <= 0 or times <= 1 or L * times > n:
        return False
    mod = (1 << 61) - 1
    base = 131542391
    h = 0
    for i in range(L):
        h = (h * base + ord(s[i])) % mod
    p = 1
    for _ in range(L - 1):
        p = (p * base) % mod
    seen = {}
    seen.setdefault(h, []).append(0)
    for i in range(1, n - L + 1):
        left_val = (ord(s[i - 1]) * p) % mod
        h = (h - left_val) % mod
        h = (h * base) % mod
        h = (h + ord(s[i + L - 1])) % mod
        if h not in seen:
            seen[h] = [i]
        else:
            seen[h].append(i)
            if len(seen[h]) >= times:
                sub_candidate = s[i:i+L]
                match_count = 0
                for start_idx in seen[h]:
                    if s[start_idx:start_idx+L] == sub_candidate:
                        match_count += 1
                        if match_count >= times:
                            return True
    return False

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type=str)
parser.add_argument('--out_path', type=str)
parser.add_argument('--out_len', type=int, default=10000)
parser.add_argument('--add_eos', action="store_true")
args = parser.parse_args()

# 输入和输出文件路径
input_file = args.in_path
output_file = args.out_path

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

def format_0shot(data):
    instruction = data['prompt'].split("<|start_header_id|>user<|end_header_id|>\n\n")[1].split("<|eot_id|>")[0]
    output = data['response']
    
    return dict(
        system = "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n",
        instruction = instruction,
        output = output.strip()  + ("<|end_of_text|>" if args.add_eos else ""),
        input = ""
    )

# 打开输入文件并逐行处理
datas = [json.loads(i) for i in open(input_file).readlines()]
random.seed(42)
random.shuffle(datas)
new_datas = []
for data in datas: 
    if has_repeated_substring_rolling_hash(data['response'], 40, 5): continue
    if has_repeated_substring_rolling_hash(data['response'], 128, 2): continue
    new_datas.append(format_0shot(data))
    
print(f"convert round2, {len(new_datas)} - sample -> {args.out_len}")
new_datas = sample_lst(new_datas, args.out_len)
with open(output_file, 'w') as f:
    f.write(json.dumps(new_datas, indent=4))

print("Processing complete. Modified data saved to", output_file)
