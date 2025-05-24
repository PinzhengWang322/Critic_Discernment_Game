import argparse
import json
from multiprocessing import Pool
from tqdm import tqdm
from extract_answer import extract_answer
from math_utils import compare_ans
import copy

def do_select(response, answer):
    response_answer = extract_answer(response)
    if ("It's a right step." not in response) and \
       ("This critic is not critical." not in response) and \
       not (compare_ans(response_answer, answer)):
        return True
    return False

critic_dic = dict()
def create_critic_data(data):
    question = data['metadata']['problem']
    wrongAnswer = data["round1_response"]
    prompt = """Your task is to evaluate a question-answer pair. 
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
"**Critic**\n\n The first mistake can be found in: 'Quoted wrong statement here.' The issue is: 'Explanation of the mistake here.
""".replace("{question}", question).replace("{wrongAnswer}", wrongAnswer)
    return {
        "instruction": prompt,
        "input": "",
        "output": critic_dic[data['critic']] 
    }

def process_data(item):
    if 'num_answer' in item['metadata']:
        item['metadata']['gold_num_answer'] = item['metadata']['num_answer']
    responses, answer, prompt = item['responses'], item['metadata']['gold_num_answer'], item['prompt']
    right_cnt = 0
    for response in responses:
        if do_select(response, answer):
            right_cnt += 1
    # print(right_cnt, len(item['responses']))
    if right_cnt / len(item['responses']) >= args.hack_ratio:
        return [create_critic_data(item)]
    else:
        return []

def write_results(path, data):
    with open(path, 'w') as f:
        f.write(json.dumps(data, indent=4))

def make_critic_dict(critic_datas):
    for critic_data in critic_datas:
        for response in critic_data["responses"]:
            critic_dic[get_critic(response)] = response
            
def get_critic(s):
    """
    提取 "The first mistake can be found in:" 后的句子内容。
    """
    # 正则表达式模式，匹配 "The first mistake can be found in:" 后的原句
    if s.count("The first mistake can be found in:") != 1: return None
    critic = "The first mistake can be found in:" + s.split("The first mistake can be found in:")[-1]
    return critic

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_critic_path', type=str)
    parser.add_argument('--in_prover_path', type=str)
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--hack_ratio', type=float)
    args = parser.parse_args()

    with open(args.in_prover_path) as f:
        datas = [json.loads(line) for line in f]

    with open(args.in_critic_path) as f:
        critic_datas = [json.loads(line) for line in f]

    make_critic_dict(critic_datas)
    results = []

    with Pool(processes=20) as pool:
        # Submit all tasks asynchronously and store the result handles
        async_results = [pool.apply_async(process_data, (data,)) for data in datas]

        # Track progress with tqdm
        for result in tqdm(async_results, total=len(async_results)):
            try:
                # Set a timeout for getting results
                local_sft_datas = result.get(timeout=5)
                results.extend(local_sft_datas)
            except TimeoutError:
                print("Timed out processing one batch of data")
            except Exception as e:
                print(f"An error occurred: {e}")

    write_results(args.out_path, results)

