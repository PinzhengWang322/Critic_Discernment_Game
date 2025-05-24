import argparse
import json
from multiprocessing import Pool, TimeoutError
from tqdm import tqdm
from extract_answer import extract_answer
from math_utils import compare_ans
import random

random.seed(42)

def is_int(s):
    """判断字符串是否能转换为 int"""
    try:
        int(s)
        return True
    except ValueError:
        return False

def process_data(item, ISGSM=False):
    responses = item['responses']
    gold_answer = extract_answer("\\boxed{" + item['gold_num_answer'] + "}")
    response_round1 = item.get("response_round1", "")

    # 第二轮答案（从 responses[0] 中取）
    response_round2 = extract_answer(responses[0], data_name="minerva_math")

    # 若第二轮答案抽取不到或不包含 "boxed"，则回退到第一轮
    if not response_round2 or "boxed" not in responses[0]:
        response_round2 = extract_answer(response_round1, data_name="minerva_math")

    # 如果是 GSM 且第二轮答案不是整数，则回退到第一轮
    if ISGSM and not is_int(response_round2):
        response_round2 = extract_answer(response_round1, data_name="minerva_math")

    # 判断对错
    round1_ans = extract_answer(response_round1, data_name="minerva_math")
    round2_ans = response_round2

    round1_correct = compare_ans(round1_ans, gold_answer) or (round1_ans.strip() == gold_answer.strip())
    round2_correct = compare_ans(round2_ans, gold_answer) or (round2_ans.strip() == gold_answer.strip())

    # 分类
    if round1_correct and round2_correct:
        return "right->right"
    elif round1_correct and not round2_correct:
        return "right->wrong"
    elif not round1_correct and round2_correct:
        return "wrong->right"
    else:
        return "wrong->wrong"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help="Path to the JSON lines file.")
    args = parser.parse_args()

    ISGSM = False  # 如果需要 GSM 逻辑，请手动置为 True

    # 读取数据
    with open(args.path, 'r', encoding='utf-8') as f:
        datas = [json.loads(line) for line in f]

    # 并行处理
    results = []
    with Pool(processes=20) as pool:
        async_results = [pool.apply_async(process_data, (data, ISGSM)) for data in datas]
        for ar in tqdm(async_results, total=len(datas)):
            try:
                results.append(ar.get(timeout=5))  # 根据需要设置超时时间
            except TimeoutError:
                print("Timed out processing one item")
            except Exception as e:
                print(f"An error occurred: {e}")

    # 统计每类结果数量
    categorized_counts = {
        "right->right": 0,
        "right->wrong": 0,
        "wrong->right": 0,
        "wrong->wrong": 0
    }
    for category in results:
        categorized_counts[category] += 1

    # 计算指标
    rr = categorized_counts["right->right"]
    rw = categorized_counts["right->wrong"]
    wr = categorized_counts["wrong->right"]
    ww = categorized_counts["wrong->wrong"]

    total = len(results)
    if total == 0:
        print("No data found!")
        exit()

    # 1. 第一轮正确的比例
    first_round_correct_ratio = (rr + rw) / total

    # 2. 在第一轮正确的样本中，第二轮改错的比例 (right->wrong 占 first_round_correct)
    first_round_correct_total = rr + rw
    if first_round_correct_total > 0:
        second_round_became_wrong_ratio = rw / first_round_correct_total
    else:
        second_round_became_wrong_ratio = 0.0

    # 3. 在第一轮错误的样本中，第二轮改对的比例 (wrong->right 占 first_round_wrong)
    first_round_wrong_total = wr + ww
    if first_round_wrong_total > 0:
        second_round_became_right_ratio = wr / first_round_wrong_total
    else:
        second_round_became_right_ratio = 0.0

    # 4. 总的正确率变化
    #    第二轮正确率 = (rr + wr) / total
    #    第一轮正确率 = (rr + rw) / total
    #    差值 = 第二轮正确率 - 第一轮正确率
    second_round_correct_ratio = (rr + wr) / total
    overall_correct_ratio_change = second_round_correct_ratio - first_round_correct_ratio

    # 输出结果
    print("============ 统计结果 ============")
    print(f"1. 第一轮正确的比例: {first_round_correct_ratio:.2%}")
    print(f"2. 在第一轮正确的样本中，第二轮改错（对->错）的比例: {second_round_became_wrong_ratio:.2%}")
    print(f"3. 在第一轮错误的样本中，第二轮改对（错->对）的比例: {second_round_became_right_ratio:.2%}")
    print(f"4. 总的正确率的变化: {overall_correct_ratio_change:.2%}")
    print("=================================")
