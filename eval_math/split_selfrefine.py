import argparse
import json
import os
from multiprocessing import Pool, TimeoutError
from tqdm import tqdm
from extract_answer import extract_answer
from math_utils import compare_ans
import random
random.seed(42)

def is_int(s):
    try:
        # 尝试将字符串转换为整数
        int(s)
        return True
    except ValueError:
        # 如果转换失败，返回 False
        return False

def process_data(item):
    responses = item['responses']
    gold_answer =  extract_answer("\\boxed{" + item['gold_num_answer'] + "}")
    prompt = item['prompt']
    response_round1 = item.get("response_round1", "")

    # First response (round 2) and initial response (round 1)
    response_round2 = extract_answer(responses[0],data_name = "minerva_math")
    if not response_round2 or "boxed" not in responses[0]:
        response_round2 = extract_answer(response_round1,data_name = "minerva_math")
    if ISGSM and not is_int(response_round2):
        response_round2 = extract_answer(response_round1,data_name = "minerva_math")

    # Evaluate responses
    round1_correct = compare_ans(extract_answer(response_round1,data_name = "minerva_math"), gold_answer)  or response_round1.strip() == gold_answer.strip()
    round2_correct = compare_ans(response_round2, gold_answer)  or response_round2.strip() == gold_answer.strip()

    # Categorize the results
    if round1_correct and round2_correct:
        category = "right->right"
    elif round1_correct and not round2_correct:
        category = "right->wrong"
    elif not round1_correct and round2_correct:
        category = "wrong->right"
    else:
        category = "wrong->wrong"

    result = {
        "category": category,
        "prompt": prompt,
        "response_round1": response_round1,
        "response_round2": responses[0],
        "gold_answer": gold_answer,
    }
    return result

def write_results(path, data):
    with open(path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    args = parser.parse_args()
    ISGSM=False
    datas = [json.loads(i) for i in open(args.path).readlines()]

    results = []
    with Pool(processes=20) as pool:
        async_results = [pool.apply_async(process_data, (data,)) for data in datas]
        for result in tqdm(async_results, total=len(datas)):
            try:
                results.append(result.get(timeout=5))  # Set timeout as needed
            except TimeoutError:
                print("Timed out processing one item")
            except Exception as e:
                print(f"An error occurred: {e}")

    # Split results by category
    categorized_results = {
        "right->right": [],
        "right->wrong": [],
        "wrong->right": [],
        "wrong->wrong": []
    }
    for res in results:
        categorized_results[res['category']].append(res)

    # Write categorized results to files
    file_name = os.path.basename(args.path)
    directory_path = os.path.dirname(args.path)

    for category, data in categorized_results.items():
        path = os.path.join(directory_path, f"{category}_{file_name}")
        write_results(path, data)

    # Print proportions
    total = len(results)
    for category, data in categorized_results.items():
        print(f"{category}: {len(data)} ({len(data) / total:.2%})")
