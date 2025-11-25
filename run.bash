#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=0,1
# ====== 路径自己改 ======
TRAIN_FILE=$PWD/data/vcbench/train.parquet
TEST_FILE=$PWD/data/vcbench/test.parquet
MODEL_PATH=Qwen/Qwen3-8B              # 或 /your/local/Qwen3-8B
REWARD_PATH=$PWD/verl/utils/reward_score/vcbench.py
REWARD_NAME=compute_score            # 你 reward 里的函数名

# ====== GRPO 训练 ======
PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
  data.train_files=$TRAIN_FILE \
  data.val_files=$TEST_FILE \
  data.train_batch_size=256 \
  data.max_prompt_length=1024 \
  data.max_response_length=256 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  \
  actor_rollout_ref.model.path=$MODEL_PATH \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=5 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  \
  custom_reward_function.path=$REWARD_PATH \
  custom_reward_function.name=$REWARD_NAME \
  \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.project_name="vcbench_grpo" \
  trainer.experiment_name="qwen3_8b" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.test_freq=10 \
  trainer.total_epochs=3 \
  2>&1 | tee grpo_vcbench_qwen3_8b.log