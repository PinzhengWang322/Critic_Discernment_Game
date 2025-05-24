import argparse
import json
import os
import numpy as np
from extract_answer import extract_answer
from math_utils import compare_ans
import random
from tqdm import tqdm
from multiprocessing import Pool, TimeoutError, Manager
random.seed(42)

# 定义超时处理的 split_data 函数
def split_data(data, counters, unique=False, timeout=5):
    responses = data['responses']
    answer = extract_answer("\\boxed{" + data['gold_num_answer'] + "}")
    prompt = data['prompt']
    meta_data = data['metadata']

    right_local, wrong_local = [], []
    random.shuffle(responses)

    try:
        for res in responses:
            # 限制每次调用 extract_answer 的时间
            response_answer = extract_answer(res)
            if compare_ans(response_answer, answer):
                meta_data["eval"] = "right"
                right_local.append({"prompt": prompt, "response": res, "answer": answer, 'metadata': meta_data})
                if unique:
                    break
            else:
                meta_data["eval"] = "wrong"
                wrong_local.append({"prompt": prompt, "response": res, "answer": answer, 'metadata': meta_data})
    except TimeoutError:
        print("Warning: Timeout occurred while processing a response.")

    with counters["lock"]:
        counters["right"].extend(right_local)
        counters["wrong"].extend(wrong_local)

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--unique', action="store_true")
    args = parser.parse_args()

    # 加载数据
    datas = [json.loads(line) for line in open(args.path).readlines()]

    # 使用 Manager 创建共享的计数器
    with Manager() as manager:
        counters = {
            "right": manager.list(),
            "wrong": manager.list(),
            "lock": manager.Lock()
        }

        # 使用多进程池并行处理数据
        with Pool(processes=10) as pool:  # 根据资源调整进程数
            async_results = [pool.apply_async(split_data, args=(data, counters, args.unique)) for data in datas]

            # 跟踪进度并处理超时
            for result in tqdm(async_results, total=len(async_results)):
                try:
                    result.get(timeout=5)  # 设置超时时间
                except TimeoutError:
                    print("Timed out processing one batch of data")
                except Exception as e:
                    print(f"An error occurred: {e}")

        # 将结果保存到文件
        file_name = os.path.basename(args.path)
        directory_path = os.path.dirname(args.path)

        right_path = os.path.join(directory_path, f"right_{file_name}")
        wrong_path = os.path.join(directory_path, f"wrong_{file_name}")

        with open(right_path, 'w') as f:
            for i in counters["right"]:
                f.write(json.dumps(i) + '\n')

        with open(wrong_path, 'w') as f:
            for i in counters["wrong"]:
                f.write(json.dumps(i) + '\n')
