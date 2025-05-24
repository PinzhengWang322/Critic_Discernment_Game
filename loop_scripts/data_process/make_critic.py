import json
import argparse
from transformers import AutoTokenizer
import os

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


def make_prompt_v2(data):
    data["metadata"]['problem'] = \
    data['prompt'].split("#### Question: ")[1].split("<|eot_id|>")[0]
    prompt = [{"role": "system", "content": "Please critic the answer carefully."}] +  \
    [{"role": "user", "content": """Your task is to evaluate a question-answer pair. 
Carefully review the question and critically assess a wrong answer.
[Question] 
```
{question}
```
      
[Wrong Answer] 
```
{wrongAnswer}
```
      
Please review the wrong answer step by step, quoting each sentence and analyzing it individually. Use the following format for your response:
Step: [Quoted Sentence]
Analysis: [Your Explanation]
Step: [Quoted Sentence]
Analysis: [Your Explanation]

After step-by-step analysis, conclude by quoting the original sentence which **first** causes the wrong answer and providing a concise yet complete explanation of the error:  
"**Critic**\n\n The first mistake can be found in: 'Quoted wrong statement here.' The issue is: 'Explanation of the mistake here.'"
""".replace("{question}", data["metadata"]["problem"]).replace("{wrongAnswer}", data['response'])}] + \
    [{"role": "assistant", "content": ""}]

    chat_prompt = tokenizer.apply_chat_template(
        prompt,
        tokenize=False
    )
    if chat_prompt.startswith(BOS_TOKEN):
        chat_prompt = chat_prompt[len(BOS_TOKEN):]
    if chat_prompt.endswith(EOS_TOKEN):
        chat_prompt = chat_prompt[:-len(EOS_TOKEN)]
    return chat_prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, required=True, help='Path to the input JSONL file')
    parser.add_argument('--out_path', type=str, required=True, help='Path to the output JSONL file')
    args = parser.parse_args()


    datas = [json.loads(i) for i in open(args.in_path).readlines()]

    new_datas = [{"idx": idx, "prompt": make_prompt_v2(data), "metadata":data['metadata'], 'round1_response':data["response"]} for idx, data in enumerate(datas)]
    with open(args.out_path, 'w') as f:
        for i in new_datas:
            f.write(json.dumps(i) + '\n')
    