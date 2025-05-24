import argparse
import json
import numpy as np
from extract_answer import get_unnormalized_answer, extract_answer
from math_utils import compare_ans
from collections import Counter
from multiprocessing import Pool, Manager, TimeoutError
from tqdm import tqdm
import random

GSM = False

random.seed(42)

def get_most_common_value_with_random_tiebreaker(responses_answer):
    counter = Counter(responses_answer)
    max_count = max(counter.values())
    most_common_values = [value for value, count in counter.items() if count == max_count]
    return most_common_values[0]

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def format(s):
    return s.replace(',', '')

def strict_check(data, counters, random_seed, timeout=5):
    random.seed(random_seed)  # Set random seed for reproducibility across runs
    responses = data['responses']
    answer = data['gold_num_answer']
    right_cnt, cnt = 0, 0

    try:
        if GSM:
            responses_answer = [format(extract_answer(res)) for res in responses if is_int(format(extract_answer(res)))]
        else:
            responses_answer = [extract_answer(res) for res in responses if "boxed" in res]
        
        if len(responses_answer) == 0:
            responses_answer = [extract_answer(res) for res in responses]
        
        random.shuffle(responses_answer)  # Shuffle the answers
        responses_answer = responses_answer[:args.m]  # Take the first 8 after shuffle
        most_common_value = get_most_common_value_with_random_tiebreaker(responses_answer)
        
        answer = extract_answer("\\boxed{" + answer + "}")
        if compare_ans(most_common_value, answer) or most_common_value.strip() == answer.strip():
            right_cnt += 1
        cnt += 1
    except TimeoutError:
        cnt += 1
        print("Warning: Timeout occurred for one response.")

    with counters["lock"]:
        counters["right_cnt"].value += right_cnt
        counters["cnt"].value += cnt

def average_results(counters, num_runs):
    # Calculate the average result for the total count and correct count
    avg_right_cnt = counters["right_cnt"].value / num_runs
    avg_cnt = counters["cnt"].value / num_runs
    
    print(f"Average total responses processed over {num_runs} runs: {avg_cnt}")
    if avg_cnt > 0:
        print(f"Average strict-match score: {avg_right_cnt / avg_cnt}")
    else:
        print("No valid data to process.")

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--num_runs', type=int, default=1, help="Number of times to repeat the experiment and take the average.")
    parser.add_argument('--m', type=int, default=64)
    args = parser.parse_args()

    # Load data
    datas = [json.loads(line) for line in open(args.path).readlines()]
    if "gsm8k" in args.path:
        GSM = True

    # Use Manager to create shared counters
    with Manager() as manager:
        counters = {
            "right_cnt": manager.Value('i', 0),
            "cnt": manager.Value('i', 0),
            "lock": manager.Lock()
        }

        num_runs = args.num_runs
        for run in range(num_runs):
            print(f"Running experiment {run + 1} of {num_runs}...")
            with Pool(processes=20) as pool:  # Adjust based on your system's resources
                async_results = [pool.apply_async(strict_check, args=(data, counters, run)) for data in datas]

                # Use a progress bar to track processing progress
                for result in tqdm(async_results, total=len(async_results)):
                    try:
                        result.get(timeout=5)  # Set timeout for each task
                    except TimeoutError:
                        print("Timed out processing one batch of data")
                    except Exception as e:
                        print(f"An error occurred: {e}")
        
        # After all runs, calculate and print the averages
        average_results(counters, num_runs)
