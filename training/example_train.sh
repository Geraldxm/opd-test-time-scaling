#!/usr/bin/env bash
set -euo pipefail

mode="${1:-print}"
[[ "$mode" == print || "$mode" == config || "$mode" == run ]] || { echo "usage: $0 [print|config|run]" >&2; exit 2; }
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
verl="$root/training/rethink_opd/verl"
data="$root/output/example_math_train.parquet"
output="${OUTPUT_DIR:-$root/output/train_example}"
python="${PYTHON_BIN:-python}"
student="${STUDENT_MODEL:-/path/to/qwen3-1.7b}"
teacher="${TEACHER_MODEL:-/path/to/qwen3-4b-teacher}"

command=("$python" -m verl.trainer.main_ppo
  algorithm.adv_estimator=token_reward_direct algorithm.grpo_outcome_weight=1.0
  data.shuffle=False "data.train_files=$data" "data.val_files=$data"
  data.train_batch_size=1 data.train_max_samples=1 data.max_prompt_length=512
  data.max_response_length=64 data.filter_overlong_prompts=True data.truncation=error
  data.return_raw_chat=True +data.apply_chat_template_kwargs.enable_thinking=False
  "actor_rollout_ref.model.path=$student" actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_activation_offload=True actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0
  actor_rollout_ref.actor.optim.lr_scheduler_type=constant actor_rollout_ref.actor.ppo_mini_batch_size=1
  actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
  actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.loss_agg_mode=token-mean
  actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
  actor_rollout_ref.actor.fsdp_config.forward_prefetch=True actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
  actor_rollout_ref.rollout.max_num_batched_tokens=32768 actor_rollout_ref.ref.fsdp_config.param_offload=True
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True +actor_rollout_ref.rollout.log_prob_top_k=16
  +actor_rollout_ref.rollout.top_k_strategy=only_stu +actor_rollout_ref.rollout.reward_weight_mode=student_p
  +actor_rollout_ref.rollout.teacher_temperature=1.0 actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 actor_rollout_ref.rollout.max_model_len=576
  actor_rollout_ref.rollout.n=1 actor_rollout_ref.rollout.repetition_penalty=1.0
  actor_rollout_ref.rollout.calculate_log_probs=True actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  reward_model.enable=True +reward_model.reward_kwargs.enable_format_reward=False
  "reward_model.model.path=$teacher" reward_model.model.input_tokenizer=null
  reward_model.model.use_remove_padding=True reward_model.model.fsdp_config.param_offload=False
  +reward_model.model.dtype=bfloat16 reward_model.micro_batch_size_per_gpu=1
  "custom_reward_function.path=$verl/verl/utils/reward_score/ttrl_math/__init__.py"
  custom_reward_function.name=reward_func trainer.val_before_train=False trainer.log_val_generations=0
  'trainer.logger=[console]' trainer.project_name=opd_example trainer.experiment_name=one_step
  trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.save_freq=1 trainer.test_freq=-1
  trainer.total_epochs=1 trainer.total_training_steps=1 trainer.resume_mode=disable
  "trainer.default_local_dir=$output/checkpoints" trainer.is_plot=False)

if [[ "$mode" == print ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

export PYTHONPATH="$verl${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$mode" == config ]]; then
  (cd "$verl" && "${command[@]}" --cfg job --resolve)
  exit 0
fi

[[ -f "$student/config.json" && -f "$teacher/config.json" ]] || {
  echo "Set STUDENT_MODEL and TEACHER_MODEL to local checkpoints" >&2; exit 1;
}
mkdir -p "$output"
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=true
(cd "$verl" && "${command[@]}")
