model_name_or_path=Qwen/Qwen3-4B-Instruct-2507
model_name_or_path=/home/t-xiaoliang/xiao/models/Qwen/Qwen2.5-0.5B-Instruct
export CUDA_VISIBLE_DEVICES=4

python scripts/eval_cot.py \
    --model_name_or_path ${model_name_or_path} \
    --input_path data/dac-rl-benchmarks.jsonl \
    --output_path outputs/evaluation-cot-style-competition.jsonl \
    --gpu_memory_utilization 0.9 \
    --top_p 0.7 \
    --temperature 1.0 \
    --n_sampling 32 \
    --tensor_parallel_size 1 \
    --max_tokens 16384  \
     --system_prompt_path prompts/gen_response.txt \
    --swap_space 64 \
    --prompt_key question 