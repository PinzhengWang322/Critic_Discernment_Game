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
    responses, answer, prompt, meta_data = item['responses'], item['gold_num_answer'], item['prompt'], item['metadata']
    temp_right = []
    temp_wrong = []
    random.shuffle(responses)
    for response in responses:
        if "boxed" not in response: continue
        response_answer = extract_answer(response)
        if answer == "a=\\frac{c-b}{x}":
            print(response_answer, answer)
            print(compare_ans(response_answer, answer) or response_answer.strip() == answer.strip())
        if (response_answer == "" or "boxed" not in response) and "response_round1" in item: 
            response_answer = extract_answer(item["response_round1"])
        if compare_ans(response_answer, answer) or response_answer.strip() == answer.strip():
            meta_data["eval"] = "right"
            temp_right.append({"prompt": prompt, "response": response, "answer": answer, 'metadata': meta_data.copy()})
            if args.unique: break
        else:
            meta_data["eval"] = "wrong"
            temp_wrong.append({"prompt": prompt, "response": response, "answer": answer, 'metadata': meta_data.copy()})
    return temp_right, temp_wrong

def write_results(path, data):
    with open(path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--unique', action="store_true")
    args = parser.parse_args()

    datas = [json.loads(i) for i in open(args.path).readlines()]

    right = []
    wrong = []
    with Pool(processes=20) as pool:
        async_results = [pool.apply_async(process_data, (data,)) for data in datas]
        for result in tqdm(async_results, total=len(datas)):
            try:
                temp_right, temp_wrong = result.get(timeout=5)  # Set timeout as needed
                right.extend(temp_right)
                wrong.extend(temp_wrong)
            except TimeoutError:
                print("Timed out processing one item")
            except Exception as e:
                print(f"An error occurred: {e}")

    file_name = os.path.basename(args.path)
    directory_path = os.path.dirname(args.path)

    unique = "unique_" if args.unique else ""
    right_path = os.path.join(directory_path, unique + f"right_{file_name}")
    wrong_path = os.path.join(directory_path, unique + f"wrong_{file_name}")

    write_results(right_path, right)
    write_results(wrong_path, wrong)
