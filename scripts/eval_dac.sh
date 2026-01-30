model_name_or_path=Qwen/Qwen3-4B-Instruct-2507
export CUDA_VISIBLE_DEVICES=4

python scripts/eval_dac.py \
    --model_name_or_path ${model_name_or_path} \
    --input_path data/dac-rl-benchmarks.jsonl \
    --output_path outputs/evaluation-dac-style-competition-d4-c8.jsonl \
    --gpu_memory_utilization 0.95 \
    --top_p 0.7 \
    --temperature 1.0 \
    --n_sampling 32 \
    --n_divide_sampling 4 \
    --n_conquer_sampling 8 \
    --tensor_parallel_size 1 \
    --max_tokens 16384  \
    --swap_space 64 \
    --prompt_key question 