import json
from dataclasses import dataclass, field
from typing import List, Optional
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests, os

tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    url: Optional[str] = field(
        default="http://localhost",
        metadata={"help": "url of the model response"},
    )
    tokenizer: Optional[str] = field(
        default="HuggingFaceH4/mistral-7b-sft-beta",
        metadata={"help": "the tokenizer to use"},
    )
    ports: List[str] = field(default_factory=lambda: ["8000"], metadata={"help": "ports of the model response"})
    eos_ids: List[int] = field(default_factory=lambda: [], metadata={"help": "the ids of the end of sentence tokens"})
    dataset_name_or_path: Optional[str] = field(
        default="cornfieldrm/iterative-prompt-v1-iter1-2K",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_path: Optional[str] = field(
        default="uf_split0_responses_K8.json",
        metadata={"help": "the location of the output file"},
    )
    bos_format: Optional[str] = field(
        default="",
        metadata={"help": "the format of the beginning of the sentence"},
    )
    K: Optional[int] = field(
        default=8,
        metadata={"help": "the number of generations per prompt"},
    )
    top_k: Optional[int] = field(
        default=-1,
        metadata={"help": "the number of generations per prompt"},
    )
    max_input_length: Optional[int] = field(
        default=10000,
        metadata={"help": "the maximum length of the input tokens"},
    )
    max_new_tokens: Optional[int] = field(
        default=4096,
        metadata={"help": "the maximum length of the new tokens"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "the random seed"},
    )
    temperature: Optional[float] = field(
        default=0.7,
        metadata={"help": "the temperature"},
    )
    use_beam_search: Optional[bool] = field(
        default=False,
        metadata={"help": "the beam search"},
    )
    dataset_key: Optional[str] = field(
        default="context_messages",
        metadata={"help": "the key of the dataset"},
    )
    max_workers: Optional[int] = field(
        default=128,
        metadata={"help": "the number of workers"},
    )
    copy_dir_depth:  Optional[int] = field(default=1)
    note: Optional[str] = field(default="")

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
ds_dir = script_args.dataset_name_or_path
K = script_args.K
ports = script_args.ports

file_name = script_args.dataset_name_or_path.split("/")[-1]

tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer)

def get_last_k_directories(path, k):
    absolute_path = os.path.abspath(path)
    path_parts = absolute_path.split(os.sep)
    if os.path.isfile(absolute_path):
        last_k_directories = path_parts[-k-1:-1]
    else:
        last_k_directories = path_parts[-k:]
    return last_k_directories

def create_path(directories):
    return  os.path.join(*directories)
    


def query_model(metadata, prompt, idx, answer, args, port):
    json = {
        **args,
        "prompt": prompt,
    }
    response = requests.post(url=script_args.url + ":" + str(port) + "/generate", json=json)

    response_json = response.json()
    return dict(
        idx = idx,
        prompt = prompt,
        responses = [response_json["text"][i][len(prompt) :] for i in range(len(response_json["text"]))],
        gold_num_answer = answer,
        response_round1 = metadata,
    )


default_args = {
    "n": script_args.K,
    "temperature": script_args.temperature,
    "max_tokens": script_args.max_new_tokens,
    "seed": script_args.seed,
    "top_p": 1.0,
    "top_k": script_args.top_k,
    "stop_token_ids": [tokenizer.eos_token_id] + script_args.eos_ids,
}


ds = [json.loads(i) for i in open(script_args.dataset_name_or_path)]

with ThreadPoolExecutor(max_workers=script_args.max_workers) as executor:
    result = [
        executor.submit(query_model, "", ds[i]["prompt"], ds[i]['idx'], ds[i]["gold_num_answer"], default_args, ports[i % len(ports)]) for i in range(len(ds))
    ]
    # use tqdm to show progress
    for _ in tqdm(as_completed(result), total=len(result)):
        pass

    responses = [r.result() for r in result]


gathered_data = []
for i in range(len(ds)):
    gathered_data.append(responses[i])

output_eval_dataset = {}
output_eval_dataset["type"] = "text_only"
output_eval_dataset["instances"] = gathered_data
print("I collect ", len(gathered_data), "samples")

file_name = script_args.output_path
dir_path = os.path.dirname(file_name)
os.makedirs(dir_path, exist_ok=True)
file_path = os.path.join(dir_path, file_name)

with open(file_path, "w", encoding="utf8") as f:
    for i in gathered_data:
        f.write(json.dumps(i, ensure_ascii=False) + '\n')
        
