import argparse
import json
from multiprocessing import Pool
from tqdm import tqdm
from extract_answer import extract_answer
from math_utils import compare_ans
import copy

def do_select(select_type, response, answer):
    if select_type == "correct":
        if "This critic is not critical." in response: return False
        response_answer = extract_answer(response)
        if compare_ans(response_answer, answer):
            return True
        else:
            return False
    elif select_type == "mislead_resist":
        if response.endswith("It's a right step.") or response.count("boxed{This critic is not critical.}") == 1:
            return True
        else:
            return False
    else:
        raise NotImplementedError(f"Type {type} is not implemented.")

def process_data(item):
    if 'num_answer' in item['metadata']:
        item['metadata']['gold_num_answer'] = item['metadata']['num_answer']
    responses, answer, prompt = item['responses'], item['metadata']['gold_num_answer'], item['prompt']
    temp_right = []
    del item['responses']
    for response in responses:
        if do_select(args.type, response, answer):
            item['response'] = response
            temp_right.append(copy.deepcopy(item))
    return temp_right

def write_results(path, data):
    with open(path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str)
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--type', type=str)
    args = parser.parse_args()

    with open(args.in_path) as f:
        datas = [json.loads(line) for line in f]

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
