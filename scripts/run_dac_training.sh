set -x

### paths and wandb settings
project_name=Your-Project-Name
logger="['console', 'wandb']"
experiment_name='Your-Experiment-Name'
model_path=Qwen/Qwen3-4B-Instruct-2507
train_path=data/DAPO-Math-17k.parquet
test_path=data/DAPO-Math-17k.parquet
save_path=outputs

### parameters
val_pass_k=True
use_chat_template=True
val_before_train=True # set to 1 to launch validation before inference
total_epochs=20
oversample_factor=8.0
tensor_model_parallel_size=1
# rollout and training batch size
validation_temperature=1.0
n_samples=8
n_competition=32
val_n_samples=1
val_batch_size=190
train_batch_size=256
ppo_mini_batch_size=64
log_prob_micro_batch_size_per_gpu=8
use_dynamic_bsz=True
save_and_test_freq=8
# sequence length
max_prompt_length=4096 
max_response_length=8192
max_validation_length=16384
ppo_max_token_len_per_gpu=20480


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${train_path} \
    data.val_files=${test_path} \
    data.train_batch_size=${train_batch_size} \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.val_max_token=${max_validation_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.val_pass_k=${val_pass_k} \
    data.val_pass_k_benchmarks="['AIME-2024-30', 'AIME-2025-30', 'Beyond-AIME-100', 'HMMT-Feb-2025']" \
    data.divide_reward_setting="any_accuracy" \
    data.conquer_reward_setting="answer_and_format" \
    data.use_chat_template=${use_chat_template} \
    data.filter_accuracy=True \
    data.oversample_factor=${oversample_factor} \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$((log_prob_micro_batch_size_per_gpu * 2)) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.val_kwargs.temperature=${validation_temperature} \
    actor_rollout_ref.rollout.val_kwargs.n_competition=${n_competition} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=${n_samples} \
    actor_rollout_ref.rollout.use_tqdm=True \
    actor_rollout_ref.rollout.val_kwargs.n=${val_n_samples} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$((log_prob_micro_batch_size_per_gpu * 2)) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=${logger} \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=${save_and_test_freq} \
    trainer.test_freq=${save_and_test_freq}  \
    trainer.total_epochs=${total_epochs} \
    trainer.val_before_train=${val_before_train} \
    trainer.default_local_dir=${save_path} \
    trainer.task='dac' $@ 