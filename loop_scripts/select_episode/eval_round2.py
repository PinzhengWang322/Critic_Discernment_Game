import argparse
import json
from multiprocessing import Pool
from tqdm import tqdm
from extract_answer import extract_answer
from math_utils import compare_ans
import copy

def do_select(select_type, response, answer):
    if select_type == "correct":
        response_answer = extract_answer(response)
        return compare_ans(response_answer, answer)
    elif select_type == "mislead_resist":
        return response.count("boxed{This critic is not critical.}") == 1
    else:
        raise NotImplementedError(f"Type {select_type} is not implemented.")

def process_data(item):
    if 'num_answer' in item['metadata']:
        item['metadata']['gold_num_answer'] = item['metadata']['num_answer']
    responses, answer = item['responses'], item['metadata']['gold_num_answer']
    success_count = sum(do_select(args.type, response, answer) for response in responses)
    return success_count, len(responses)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, required=True, help="Input file path")
    parser.add_argument('--type', type=str, required=True, choices=["correct", "mislead_resist"], help="Evaluation type")
    args = parser.parse_args()

    with open(args.in_path) as f:
        datas = [json.loads(line) for line in f][:200]

    total_success = 0
    total_responses = 0

    with Pool(processes=20) as pool:
        async_results = [pool.apply_async(process_data, (data,)) for data in datas]

        for result in tqdm(async_results, total=len(async_results)):
            try:
                success_count, response_count = result.get(timeout=5)
                total_success += success_count
                total_responses += response_count
            except TimeoutError:
                print("Timed out processing one batch of data")
            except Exception as e:
                print(f"An error occurred: {e}")

    success_rate = (total_success / total_responses) * 100 if total_responses > 0 else 0
    print(f"Success rate for '{args.type}': {success_rate:.2f}% ({total_success}/{total_responses})")
