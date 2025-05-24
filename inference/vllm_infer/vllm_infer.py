import argparse
import json
import multiprocessing as mp
import os
from tqdm import tqdm
from transformers import AutoTokenizer

def worker(gpu_id, prompts_chunk, original_data_chunk, args, return_list):
    import os
    from vllm import LLM, SamplingParams

    # Set GPU environment
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Initialize the LLM model
    llm = LLM(model=args.model_path, gpu_memory_utilization=0.9, swap_space=64, max_num_seqs=256, enforce_eager=True)

    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=4096,
        n=args.k,
        top_p=0.95,
        top_k=5,
        stop=['<|eot_id|>', '<|end_of_text|>', '<im_end>'],
        stop_token_ids=[128009]
    )

    # Split prompts_chunk and original_data_chunk into batches of size 256
    batch_size = 128
    num_batches = (len(prompts_chunk) + batch_size - 1) // batch_size

    results = []

    # Use tqdm for batch-level progress bar
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(prompts_chunk))

        # Get the current batch
        batch_prompts = prompts_chunk[batch_start:batch_end]
        batch_original_data = original_data_chunk[batch_start:batch_end]

        # Generate responses for the current batch
        outputs = llm.generate(batch_prompts, sampling_params=sampling_params, use_tqdm=False)

        for i, output in enumerate(outputs):
            generations = [o.text for o in output.outputs]
            original_prompt_data = batch_original_data[i]  # Get original data for this prompt
            original_prompt_data['responses'] = generations  # Insert generated responses
            results.append(original_prompt_data)

    # Extend the return list with the results
    return_list.extend(results)

def main():
    parser = argparse.ArgumentParser(description="vllm inference")
    parser.add_argument('--jsonl_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=8)
    args = parser.parse_args()

    parent_dir = os.path.dirname(args.out_path)
    os.makedirs(parent_dir, exist_ok=True)
    
    prompts = []
    original_data = []
    large_prompts = []
    tokenizer_path = os.environ['BASE_MODEL_NAME']
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    line_datas = open(args.jsonl_path, 'r', encoding='utf-8').readlines()
    import random; random.seed(42)
    random.shuffle(line_datas)
    for line in line_datas:
        data = json.loads(line.strip())
        if len(tokenizer(data['prompt'])['input_ids']) > 8000:
            data['responses'] = [""]  # Add an empty 'responses' field
            large_prompts.append(data)
        else:
            prompts.append(data['prompt'])
            original_data.append(data)

    num_gpus = args.num_gpus
    chunk_size = (len(prompts) + num_gpus - 1) // num_gpus
    prompts_chunks = [prompts[i*chunk_size:(i+1)*chunk_size] for i in range(num_gpus)]
    original_data_chunks = [original_data[i*chunk_size:(i+1)*chunk_size] for i in range(num_gpus)]

    manager = mp.Manager()
    return_list = manager.list()
    processes = []

    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker, args=(gpu_id, prompts_chunks[gpu_id], original_data_chunks[gpu_id], args, return_list))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


    with open(args.out_path, 'w', encoding='utf-8') as f_out:
        for result in return_list:
            f_out.write(json.dumps(result, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    main()
