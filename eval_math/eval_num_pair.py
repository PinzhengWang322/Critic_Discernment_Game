import argparse
import json
import numpy as np
from extract_answer import get_unnormalized_answer, extract_answer
from math_utils import compare_ans
from tqdm import tqdm
from multiprocessing import Pool, Manager, TimeoutError

# 定义用于检查答案的函数
def strict_check(data_a, data_b, counters, timeout=5):
    responses_a = data_a['responses']
    responses_b = data_b['responses']
    answer_a = data_a['gold_num_answer']
    answer_b = data_b['gold_num_answer']
    
    right_cnt_a, cnt_a = 0, 0
    right_cnt_b, cnt_b = 0, 0

    try:
        for res_a, res_b in zip(responses_a, responses_b):
            # 检查数据中是否包含 "boxed"
            if "boxed" in res_a and "boxed" in res_b:
                response_answer_a = extract_answer(res_a, data_name=args.data_name)
                response_answer_b = extract_answer(res_b, data_name=args.data_name)

                if (response_answer_a == "" or "It's a right step." in res_a) and "response_round1" in data_a:
                    response_answer_a = extract_answer(data_a["response_round1"], data_name=args.data_name)
                if (response_answer_b == "" or "It's a right step." in res_b) and "response_round1" in data_b:
                    response_answer_b = extract_answer(data_b["response_round1"], data_name=args.data_name)

                extracted_answer_a = extract_answer("\\boxed{" + answer_a + "}")
                extracted_answer_b = extract_answer("\\boxed{" + answer_b + "}")

                if compare_ans(response_answer_a, extracted_answer_a) or response_answer_a.strip() == extracted_answer_a.strip():
                    right_cnt_a += 1
                if compare_ans(response_answer_b, extracted_answer_b) or response_answer_b.strip() == extracted_answer_b.strip():
                    right_cnt_b += 1

                cnt_a += 1
                cnt_b += 1
    except TimeoutError:
        cnt_a += 1
        cnt_b += 1
        print("Warning: Timeout occurred for one response.")

    with counters["lock"]:
        counters["right_cnt_a"].value += right_cnt_a
        counters["cnt_a"].value += cnt_a
        counters["right_cnt_b"].value += right_cnt_b
        counters["cnt_b"].value += cnt_b

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_a', type=str, required=True, help="Path to the first JSONL file.")
    parser.add_argument('--path_b', type=str, required=True, help="Path to the second JSONL file.")
    parser.add_argument('--data_name', type=str, default="minerva_math", help="Data name for answer extraction.")
    args = parser.parse_args()

    # 加载数据
    datas_a = [json.loads(line) for line in open(args.path_a).readlines()]
    datas_b = [json.loads(line) for line in open(args.path_b).readlines()]

    if len(datas_a) != len(datas_b):
        raise ValueError("The two JSONL files must have the same number of lines.")

    # 使用 Manager 创建共享计数器
    with Manager() as manager:
        counters = {
            "right_cnt_a": manager.Value('i', 0),
            "cnt_a": manager.Value('i', 0),
            "right_cnt_b": manager.Value('i', 0),
            "cnt_b": manager.Value('i', 0),
            "lock": manager.Lock()
        }

        # 使用多进程池并行处理数据
        with Pool(processes=20) as pool:  # 根据系统资源调整进程数量
            async_results = [
                pool.apply_async(strict_check, args=(data_a, data_b, counters))
                for data_a, data_b in zip(datas_a, datas_b)
            ]

            # 进度条跟踪处理进度
            for result in tqdm(async_results, total=len(async_results)):
                try:
                    result.get(timeout=5)  # 设置超时时间
                except TimeoutError:
                    print("Timed out processing one batch of data")
                except Exception as e:
                    print(f"An error occurred: {e}")

        # 输出结果
        if counters["cnt_a"].value > 0:
            print("File A strict-match score:", counters["right_cnt_a"].value / counters["cnt_a"].value, " ,num:", counters["cnt_a"].value)
        else:
            print("No valid data to process in file A.")

        if counters["cnt_b"].value > 0:
            print("File B strict-match score:", counters["right_cnt_b"].value / counters["cnt_b"].value, " ,num:", counters["cnt_b"].value)
        else:
            print("No valid data to process in file B.")
