import argparse
import json
import random
from collections import Counter
from nltk.util import ngrams

def calculate_4gram_repetition_rate(text):
    n = 4
    tokens = list(text)
    fourgrams = list(ngrams(tokens, n))
    fourgram_counter = Counter(fourgrams)
    total_fourgrams = sum(fourgram_counter.values())
    unique_fourgrams = len(fourgram_counter)
    if total_fourgrams == 0:
        return 0.0
    repetition_rate = (total_fourgrams - unique_fourgrams) / total_fourgrams
    return repetition_rate

def main():
    parser = argparse.ArgumentParser(description='分析JSONL文件中的文本重复度等指标')
    parser.add_argument('--path', help='输入的jsonl文件路径')
    args = parser.parse_args()

    total_length = 0
    total_repetition_rate = 0.0
    total_however_count = 0
    num_texts = 0

    with open(args.path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if 'response' in data:
                text = data['response']
            elif 'responses' in data and isinstance(data['responses'], list):
                text = random.choice(data['responses'])
            else:
                continue  # 如果既没有'response'也没有'responses'，跳过此行

            num_texts += 1
            text_length = len(text)
            total_length += text_length

            # 计算4-gram重复度
            repetition_rate = calculate_4gram_repetition_rate(text)
            total_repetition_rate += repetition_rate

            # 统计出现'However'的次数
            however_count = text.count('However')
            total_however_count += however_count

    if num_texts == 0:
        print('未找到有效的文本数据进行分析。')
        return

    average_length = total_length / num_texts
    average_repetition_rate = total_repetition_rate / num_texts
    average_however_count = total_however_count / num_texts

    print(f'文本数量: {num_texts}')
    print(f'字符串平均长度: {average_length:.2f}')
    print(f'字符串平均4-gram重复度: {average_repetition_rate:.2%}')
    print(f'字符串中平均出现"However"的次数: {average_however_count:.2f}')

if __name__ == '__main__':
    main()
