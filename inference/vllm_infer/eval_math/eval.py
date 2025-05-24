import argparse
import json
import numpy as np
from extract_answer import get_unnormalized_answer, extract_answer
from math_utils import compare_ans
from tqdm import tqdm
from multiprocessing import Pool, Manager, TimeoutError


def strict_check(data, counters, timeout=5):
    responses = data['responses']
    answer = data['gold_num_answer']
    right_cnt, cnt = 0, 0

    try:
        for res in responses:
            response_answer = extract_answer(res, data_name = args.data_name)
            if (response_answer == "" or "boxed" not in res) and "response_round1" in data or "answer is correct" in res: 
                response_answer = extract_answer(data["response_round1"], data_name = args.data_name)
            answer = extract_answer("\\boxed{" + answer+ "}")
            if compare_ans(response_answer, answer) or response_answer.strip() == answer.strip():
                right_cnt += 1
            cnt += 1
    except TimeoutError:
        cnt += 1
        print("Warning: Timeout occurred for one response.")

    with counters["lock"]:
        counters["right_cnt"].value += right_cnt
        counters["cnt"].value += cnt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--data_name', type=str, default="minerva_math")
    args = parser.parse_args()


    datas = [json.loads(line) for line in open(args.path).readlines()]


    with Manager() as manager:
        counters = {
            "right_cnt": manager.Value('i', 0),
            "cnt": manager.Value('i', 0),
            "lock": manager.Lock()
        }


        with Pool(processes=20) as pool: 
            async_results = [pool.apply_async(strict_check, args=(data, counters)) for data in datas]


            for result in tqdm(async_results, total=len(async_results)):
                try:
                    result.get(timeout=5)  
                except TimeoutError:
                    print("Timed out processing one batch of data")
                except Exception as e:
                    print(f"An error occurred: {e}")

        if counters["cnt"].value > 0:
            print("strict-match score:", counters["right_cnt"].value / counters["cnt"].value)
        else:
            print("No valid data to process.")
