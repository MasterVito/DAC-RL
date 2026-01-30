import os
import re
import sys
import json
import torch
import argparse
import numpy as np
from datasets import Dataset as HFDataset
from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import AutoTokenizer
sys.path.append(os.getcwd())
from verl.utils.reward_score.prime_math import compute_score
NUM_PROC = os.cpu_count() // 2


def calculate_accuracy(sample):
    '''Getting accuracy from predictions'''
    accs = []
    for pred in sample['prediction']:
        print(f"Prediction: {pred} | Reference: {sample['ref_answer']}")
        acc_compute_score = compute_score(pred, sample['ref_answer'])
        acc = 1.0 if acc_compute_score else 0.0
        accs.append(acc)
    return {"acc": np.sum(accs), "if_acc": accs}


def format_divide_batch(batch, tokenizer):
    question_strs = np.array(batch['vllm_input']).tolist() 
    divide_input = [tokenizer.apply_chat_template([{"content": divide_prompt.replace(r"{REPLACE}", q), "role": "user"}], tokenize=False, add_generation_prompt=True) for q in question_strs]
    return divide_input


def format_conquer_batch(batch, args=None):
    def getting_subproblems(sample):
        pred = sample['response']
        pred_format_correct = True
        subproblems = re.findall(r'<SUBPROBLEM \d+>(.*?)</SUBPROBLEM \d+>', pred, re.DOTALL)
        
        if len(subproblems) == 0:
            pred_format_correct = False # format is incorrect if subproblems are not located in the expected format
            matches = list(re.finditer(r'<SUBPROBLEM (\d+)>', pred))
            subproblems = []

            for i, match in enumerate(matches):
                start_idx = match.end() 
                if i + 1 < len(matches):
                    end_idx = matches[i + 1].start()  
                else:
                    end_idx = len(pred)  
                content = pred[start_idx:end_idx].strip()
                subproblems.append(content.strip("\n"))
        
        else:
            for i in range(len(subproblems)):
                if not re.search(r'<SUBPROBLEM {}>'.format(i+1), pred) or not re.search(r'</SUBPROBLEM {}>'.format(i+1), pred):
                    pred_format_correct = False # format is incorrect if subproblem tags are missing
                    break
                
        if len(subproblems) == 0:
            subproblems = [""]
        return {'subproblem_counts': len(subproblems), 'subproblems': subproblems, 'format_correct': pred_format_correct, 'number_correct': len(subproblems) >= 3}

    def forming_subproblems_to_conquer_prompt(sample):
        if sample['subproblems'] == [""]:
            sample['subproblems'] = []
        subproblems = [x.strip("\n") for x in sample['subproblems']]
        subproblem_string = "\n\n".join([f"<SUBPROBLEM {i+1}>\n{subproblems[i].strip()}\n</SUBPROBLEM {i+1}>" for i in range(len(subproblems))])
        final_prompt = conquer_prompt.replace("{Subproblems}", subproblem_string).replace(r"{Original}", sample['question'].strip())
        sample['conquer_prompt'] = final_prompt
        sample['used_subproblem_counts'] = len(subproblems)
        return sample


    divide_questions = np.array(batch['vllm_input']).tolist()
    divide_questions_expand = [x for x in divide_questions for _ in range(args.n_divide_sampling)]
    divide_subproblems = [x for sublist in np.array(batch['divide_output']).tolist() for x in sublist]
    print(f"Length of divide questions: {len(divide_questions_expand)}, Length of divide subproblems: {len(divide_subproblems)}, N_divide_sampling: {args.n_divide_sampling}, N_conquer_sampling: {args.n_conquer_sampling}, N_sampling: {args.n_sampling}")
    divide_responses_dataset = HFDataset.from_dict({"response": divide_subproblems, "question": divide_questions_expand})
    divide_responses_dataset = divide_responses_dataset.map(getting_subproblems, num_proc=max(1, os.cpu_count() // 4))
    print(f"Average Subproblems: {np.mean(divide_responses_dataset['subproblem_counts'])}, Max Subproblems: {np.max(divide_responses_dataset['subproblem_counts'])}, Format Correct Rate: {np.mean(divide_responses_dataset['format_correct'])}, Number Correct Rate (>=3): {np.mean(divide_responses_dataset['number_correct'])}")
    divide_responses_dataset = divide_responses_dataset.map(forming_subproblems_to_conquer_prompt, num_proc=max(1, os.cpu_count() // 4))
    return np.array(divide_responses_dataset['conquer_prompt']).tolist()


def evaluate_and_print(outputs, model_path, additional_str=""):
    # loading the dataset
    result_path = os.path.join(model_path, f"eval_math_all_0327{additional_str}_results.json")
    if os.path.exists(result_path):
        split_accs = json.load(open(result_path, "r"))
        print(split_accs)
        return split_accs
    
    # verify the responses and statistics the data_sources
    split_accs = {"Model": model_path.split("/")[-1]}
    outputs_verify = outputs.map(calculate_accuracy, num_proc=64)
    data_sources = outputs_verify.unique('data_source')
    for data_source in sorted(data_sources):
        subset = outputs_verify.filter(lambda x:x["data_source"] in [data_source], num_proc=NUM_PROC)
        split_accs[data_source] = round(np.mean(subset['acc']).item() * 100 / len(outputs[0]["prediction"]), 2)
        
    # calculate the average and save the results
    split_accs['AVG'] = round(np.mean([v for k, v in split_accs.items() if k not in ['Model']]).item(), 2)
    print(split_accs)
    json.dump(split_accs, open(result_path, "w"))
    return split_accs, outputs_verify


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--prompt_key", default='prompt', type=str)
    parser.add_argument("--tensor_parallel_size", default=1, type=int, choices=[1, 2, 4, 8])
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--n_divide_sampling", default=-1, type=int, help="Number of groups of divided subproblems. Set to -1 to use n_sampling")
    parser.add_argument("--n_conquer_sampling", default=-1, type=int, help="Number of groups of conquer solutions per subproblem. Set to -1 to use n_sampling")
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--top_k", default=-1, type=int)
    parser.add_argument("--gpu_memory_utilization", default=1.0, type=float) 
    parser.add_argument("--swap_space", default=4, type=int)
    parser.add_argument("--max_tokens", default=512, type=int)
    parser.add_argument("--max_model_len", default=32768, type=int, help="The maximum length of the model input, need to set after vLLM 0.9.1.")
    parser.add_argument("--total_chunk", default=-1, type=int, help="Set to >0 to split the dataset to small chunks.")
    parser.add_argument("--current_chunk", default=-1, type=int, help="Set to >0 to choose the current chunk.")
    args = parser.parse_args()
    available_gpus = torch.cuda.device_count()
    
    if args.n_divide_sampling > 0 and args.n_conquer_sampling > 0 and args.n_conquer_sampling * args.n_divide_sampling == args.n_sampling:
        sample_params_divide = SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, n=args.n_divide_sampling, max_tokens=args.max_tokens, stop=["<end>", "\] \] \]", "\n\n\n"])
        sample_params_conquer = SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, n=args.n_conquer_sampling, max_tokens=args.max_tokens, stop=["<end>", "\] \] \]", "\n\n\n"])
    else:
        args.n_divide_sampling = 1
        args.n_conquer_sampling = args.n_sampling
        sample_params_divide = SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, n=1, max_tokens=args.max_tokens, stop=["<end>", "\] \] \]", "\n\n\n"])
        sample_params_conquer = SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, n=args.n_sampling, max_tokens=args.max_tokens, stop=["<end>", "\] \] \]", "\n\n\n"])
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    divide_prompt = open("./prompts/prompt_for_divide.txt", "r").read()
    conquer_prompt = open("./prompts/prompt_for_conquer.txt", "r").read()
    
    dataset = load_dataset("json", data_files=args.input_path)["train"]
    dataset = dataset.map(lambda x: {"vllm_input": x[args.prompt_key]}, num_proc=64)
    dataset = dataset.add_column("divide_prompt", format_divide_batch(dataset, tokenizer))
    print(f">>> Dataset Keys: {dataset.column_names}, Total Samples: {len(dataset)}")
    
    ### start inference
    print(f">>> Starting inference...")
    model = LLM(args.model_name_or_path, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization, swap_space=args.swap_space, max_model_len=args.max_model_len)
    outputs = model.generate(dataset["divide_prompt"], sample_params_divide)
    outputs = [[_.outputs[i].text.split("<|endoftext|>")[0] for i in range(len(_.outputs))] for _ in outputs]

    ### conquer
    dataset = dataset.add_column("divide_output", outputs)
    # dataset = dataset.add_column("conquer_prompt", format_conquer_batch(dataset, args))
    conquer_prompts = format_conquer_batch(dataset, args)
    assert len(conquer_prompts) == len(dataset) * args.n_divide_sampling, "Length of conquer prompts does not match the expected number."
    print(f">>> Got {len(conquer_prompts)} conquer prompts.")
    dataset = dataset.add_column("conquer_prompt", [conquer_prompts[i:i+args.n_divide_sampling] for i in range(0, len(conquer_prompts), args.n_divide_sampling)])
    outputs = model.generate(conquer_prompts, sample_params_conquer)

    outputs = [[_.outputs[i].text.split("<|endoftext|>")[0] for i in range(len(_.outputs))] for _ in outputs]
    outputs_expand = [x for sublist in outputs for x in sublist]
    assert len(outputs_expand) == len(dataset) * args.n_divide_sampling * args.n_conquer_sampling, "Length of conquer outputs does not match the expected number."
    print(f">>> Got {len(outputs_expand)} conquer solutions.")
    outputs_contract = [outputs_expand[i:i+args.n_sampling] for i in range(0, len(outputs_expand), args.n_sampling)]
    print(f">>> Finishing inference")
    print(f">>> Dataset Keys: {dataset.column_names}")
    dataset = dataset.add_column("prediction", outputs_contract)
    dataset = dataset.remove_columns(["vllm_input"])

    res, dataset = evaluate_and_print(dataset, args.model_name_or_path, additional_str=f"_sample_{len(dataset)}_temp_{args.temperature}_n_{args.n_sampling}")
    dataset.to_json(args.output_path)
    
