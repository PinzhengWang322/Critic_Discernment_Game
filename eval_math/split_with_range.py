import argparse
import json, os
from multiprocessing import Pool, TimeoutError
from extract_answer import extract_answer
from math_utils import compare_ans
from tqdm import tqdm
import random

def is_prove(s):
    return ("prove" in s.lower()) or ("show" in s.lower())

def is_multi_choice(gold_s):
    for i in ["A","B","C","D","E"]:
        if gold_s.strip() == i:
            return True
    return False

def is_synthetic(data):
    return "synthetic" in data['source']

def process_data(item, unique=False):
    responses, answer, prompt, gold_answer, meta_data = item['responses'], item['gold_num_answer'], item['prompt'], item['gold_answer'], item['meta_data']
    temp_right = []
    temp_wrong = []
    correct_count = 0

    random.shuffle(responses)
    responses_answer = [extract_answer(res) for res in responses]
    low_threshold = args.low_choice_threshold if is_multi_choice(answer) else args.low_threshold
    high_threshold = args.high_threshold

    meta_data['gold_answer'] = gold_answer
    if not (is_prove(meta_data['problem']) or is_synthetic(meta_data)):
        for i, response in zip(responses_answer, responses):
            if compare_ans(i, answer):
                correct_count += 1
                if unique and len(temp_right) == 1: continue
                temp_right.append({"prompt": prompt, "response": response, "gold_num_answer": answer, 'metadata': meta_data})
            else:
                if unique and len(temp_wrong) == 1: continue
                temp_wrong.append({"prompt": prompt, "response": response, "gold_num_answer": answer, 'metadata': meta_data})
    else:
        correct_count = -1

    if low_threshold <= correct_count <= high_threshold:
        for i in temp_right: i['metadata']["correct_count"] = correct_count
        for i in temp_wrong: i['metadata']["correct_count"] = correct_count
        return temp_right, temp_wrong, correct_count
    else:
        return [], [], correct_count
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--low_threshold', type=int, default=1)
    parser.add_argument('--low_choice_threshold', type=int, default=4)
    parser.add_argument('--high_threshold', type=int, default=6)
    parser.add_argument('--unique', action="store_true")
    parser.add_argument('--out_dir', type=str)
    args = parser.parse_args()
    random.seed(44)

    datas = [json.loads(line) for line in open(args.path).readlines()]
    right = []
    wrong = []
    response_count_statistics = {}

    # process_data(datas[0], args.unique)
    with Pool(processes=20) as pool:
        async_results = [pool.apply_async(process_data, (data, args.unique)) for data in datas]
        for result in tqdm(async_results, total=len(async_results)):
            try:
                temp_right, temp_wrong, correct_count = result.get(timeout=5)
                right.extend(temp_right)
                wrong.extend(temp_wrong)
                if correct_count in response_count_statistics:
                    response_count_statistics[correct_count] += 1
                else:
                    response_count_statistics[correct_count] = 1
            except TimeoutError:
                print("Timed out processing one batch of data")
            except Exception as e:
                print(f"An error occurred: {e}")

    file_name = os.path.basename(args.path)
    directory_path = args.out_dir

    right_path = os.path.join(directory_path, f"right_{args.low_threshold}_{args.high_threshold}"+ ("_unique" if args.unique else "") +  f"_{file_name}")
    wrong_path = os.path.join(directory_path, f"wrong_{args.low_threshold}_{args.high_threshold}"+ ("_unique" if args.unique else "") +  f"_{file_name}")

    with open(right_path, 'w') as f:
        for i in right:
            f.write(json.dumps(i) + '\n')

    with open(wrong_path, 'w') as f:
        for i in wrong:
            f.write(json.dumps(i) + '\n')

    print("Response Correct Count Statistics:")
    for count, num_questions in response_count_statistics.items():
        print(f"Correct answers for {count} responses: {num_questions}")
