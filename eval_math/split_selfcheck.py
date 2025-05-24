import argparse
import json
import os
from multiprocessing import Pool, TimeoutError
from tqdm import tqdm
from extract_answer import extract_answer
from math_utils import compare_ans
import random

random.seed(42)

def process_data(item):
    responses = item['responses']
    if 'gold_num_answer' in item['metadata']:
        item['gold_num_answer'] = item['metadata']['gold_num_answer']
    if 'num_answer' in item['metadata']:
        item['gold_num_answer'] = item['metadata']['num_answer']
    gold_answer = item['gold_num_answer']
    prompt = item['prompt']
    response_round1 = item.get("response_round1", "")

    # Extract and evaluate responses
    round1_extracted = extract_answer(response_round1)
    round1_correct = compare_ans(round1_extracted, gold_answer)

    valid_round2_responses = []
    for response in responses:
        extracted = extract_answer(response)
        if compare_ans(extracted, gold_answer):
            valid_round2_responses.append(response)

    # Randomly select a valid response if any
    if valid_round2_responses:
        selected_response = random.choice(valid_round2_responses)
    else:
        selected_response = None

    round2_correct = selected_response is not None

    # Categorize the results
    if round1_correct and round2_correct:
        category = "right->right"
    elif not round1_correct and round2_correct:
        category = "wrong->right"
    else:
        return None  # Skip entries not in the target categories

    result = {
        "category": category,
        "prompt": prompt,
        "response_round1": response_round1,
        "response_round2": selected_response,
        "gold_answer": gold_answer,
        "response": selected_response
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

    datas = [json.loads(i) for i in open(args.path).readlines()]

    results = []
    with Pool(processes=20) as pool:
        async_results = [pool.apply_async(process_data, (data,)) for data in datas]
        for result in tqdm(async_results, total=len(datas)):
            try:
                processed = result.get(timeout=5)  # Set timeout as needed
                if processed:
                    results.append(processed)
            except TimeoutError:
                print("Timed out processing one item")
            except Exception as e:
                print(f"An error occurred: {e}")

    # Split results by category
    categorized_results = {
        "right->right": [],
        "wrong->right": []
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

"""
python split_selfcheck.py \
--path /online1/ycsc_lijt1/lijt1/wpz/Llama_CDG/dataset/self-check/outputs/instruct_right_train.jsonl; \
python split_selfcheck.py \
--path /online1/ycsc_lijt1/lijt1/wpz/Llama_CDG/dataset/self-check/outputs/instruct_wrong_train.jsonl; \
python split_selfcheck.py \
--path /online1/ycsc_lijt1/lijt1/wpz/Llama_CDG/dataset/self-check/outputs/V1.2.2_right_train.jsonl; \
python split_selfcheck.py \
--path /online1/ycsc_lijt1/lijt1/wpz/Llama_CDG/dataset/self-check/outputs/V1.2.2_wrong_train.jsonl
"""
