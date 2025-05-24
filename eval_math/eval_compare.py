import argparse
import json
import hashlib
from extract_answer import extract_answer
from math_utils import compare_ans
from tqdm import tqdm

# 定义用于检查答案的函数
def check_answer(data_a, data_b, data_name):
    """
    检查在 file_a 中正确但在 file_b 中错误的例子。
    """
    result = {
        "question": data_a["question"],
        "gold_answer": data_a["gold_num_answer"],
        "responses_a": data_a["responses"],
        "responses_b": data_b["responses"],
    }

    try:
        # 提取正确答案
        gold_answer = extract_answer("\\boxed{" + data_a["gold_num_answer"] + "}")

        # 检查 file_a 中的正确性
        correct_in_a = any(
            compare_ans(extract_answer(res, data_name=data_name), gold_answer) or
            extract_answer(res, data_name=data_name).strip() == gold_answer.strip()
            for res in data_a["responses"]
        )

        # 检查 file_b 中的正确性
        correct_in_b = any(
            compare_ans(extract_answer(res, data_name=data_name), gold_answer) or
            extract_answer(res, data_name=data_name).strip() == gold_answer.strip()
            for res in data_b["responses"]
        )

        # 如果 a 正确且 b 错误，返回此结果
        if correct_in_a and not correct_in_b:
            return result
    except Exception as e:
        print(f"Error processing question: {data_a['question']}, error: {e}")

    return None

def compute_hash(data):
    """Compute a hash for a question to ensure matching."""
    return hashlib.md5(data["question"].encode('utf-8')).hexdigest()

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_a', type=str, required=True, help="Path to the first input JSONL file")
    parser.add_argument('--file_b', type=str, required=True, help="Path to the second input JSONL file")
    parser.add_argument('--out_path', type=str, required=True, help="Path to the output JSONL file")
    parser.add_argument('--data_name', type=str, default="minerva_math", help="Dataset name")
    args = parser.parse_args()

    # 加载数据
    with open(args.file_a, 'r') as f_a, open(args.file_b, 'r') as f_b:
        datas_a = [json.loads(line) for line in f_a.readlines()]
        datas_b = [json.loads(line) for line in f_b.readlines()]

    # 构建 hash -> data 映射
    hash_to_data_a = {compute_hash(data): data for data in datas_a}
    hash_to_data_b = {compute_hash(data): data for data in datas_b}

    results = []

    # 对每对匹配数据进行处理
    for hash_key, data_a in tqdm(hash_to_data_a.items()):
        data_b = hash_to_data_b.get(hash_key)
        if data_b:
            result = check_answer(data_a, data_b, args.data_name)
            if result:
                results.append(result)

    # 将结果写入输出文件
    with open(args.out_path, 'w') as out_file:
        for result in results:
            out_file.write(json.dumps(result) + "\n")

    print(f"Processing complete. Results saved to {args.out_path}")
