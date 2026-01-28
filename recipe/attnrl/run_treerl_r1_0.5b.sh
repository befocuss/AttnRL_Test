#!/usr/bin/env bash

set -x

set -euo pipefail

# Optional overrides:
# - NEW_CONDA_HOME: conda install prefix (e.g., /home/username/miniforge3)
# - BASE_PATH: AttnRL repo root (e.g., /home/username/AttnRL)
# - CONDA_ENV_NAME: conda env name (default: attnrl)
NEW_CONDA_HOME="${NEW_CONDA_HOME:-}"
BASE_PATH="${BASE_PATH:-}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-attnrl}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${BASE_PATH}" ]]; then
  # This script lives at: AttnRL/recipe/attnrl/*.sh → repo root is two levels up.
  BASE_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

# Prefer an existing `conda` on PATH; otherwise fall back to NEW_CONDA_HOME.
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [[ -n "${NEW_CONDA_HOME}" && -x "${NEW_CONDA_HOME}/bin/conda" ]]; then
  CONDA_BASE="${NEW_CONDA_HOME}"
else
  echo "ERROR: conda not found. Ensure conda is on PATH, or set NEW_CONDA_HOME to your conda prefix." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
PYTHON_BIN="${CONDA_PREFIX}/bin/python"

MODEL_PATH=/root/Model/Qwen2.5-7B-Instruct
wandb_offline=False

# cp $BASE_PATH/recipe/modeling_qwen2.py $NEW_CONDA_HOME/envs/attnrl/lib/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py
cd $BASE_PATH

export WANDB_API_KEY=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

project_name='AttnRL'
exp_name='Qwen2.5-VL-7B-Instruct-TreeRL-Math'

adv_estimator=attnrl
use_kl_in_reward=False
kl_coef=0.001
use_kl_loss=True
kl_loss_coef=0.001
clip_ratio_low=0.2
clip_ratio_high=0.28
loss_agg_mode="token-mean"

max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 1))

# Further reduced batch size for 7B model (OOM fix - step 2)
train_prompt_bsz=2  # Reduced from 4 to 2
n_resp_per_prompt=2
train_prompt_mini_bsz=1  # Reduced from 2 to 1

# Ray
# RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
# WORKING_DIR=${WORKING_DIR:-"${PWD}"}
# RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}
# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${BASE_PATH}"}
CKPTS_DIR=${CKPTS_DIR:-"/opt/mizar"}
ROLLOUT_DIR=${VAL_DIR:-"${RAY_DATA_HOME}/rollout/${project_name}/${exp_name}"}
VAL_DIR=${VAL_DIR:-"${RAY_DATA_HOME}/val/${project_name}/${exp_name}"}
REWARD_DIR=${REWARD_DIR:-"${RAY_DATA_HOME}/reward/${project_name}/${exp_name}"}
WANDB_DIR_CUSTOM=${WANDB_DIR_CUSTOM:-"${RAY_DATA_HOME}/wandb_dir/${project_name}/${exp_name}"}
LOG_DIR=${LOG_DIR:-"${RAY_DATA_HOME}/logs/${project_name}/${exp_name}"}
mkdir -p ${LOG_DIR}
# Math level 3-5 dataset (Parquet format)
TRAIN_FILE=${TRAIN_FILE:-"/root/Data/math_oat/math_lvl3to5_8k_parquet/train.parquet"}
# Math level 3-5 eval set
TEST_FILE=${TEST_FILE:-"/root/Data/math_oat/math_lvl3to5_8k_parquet/eval.parquet"}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1
val_temperature=0
val_top_p=1.0

# Performance Related Parameter - Optimized for 7B model with memory constraints
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
max_num_batched_tokens=$(((max_prompt_length + max_response_length) * 1))
max_model_len=$((max_prompt_length + max_response_length * 1))
max_num_seqs=$((256))  # Reduced from 512 to 256 for OOM fix
offload=True  # Enable offloading to save GPU memory
gen_tp=1
fsdp_size=-1
free_cache_engine=True
disable_log_stats=True
max_actor_ckpt_to_keep=1

balance_val_batch=True
balance_gen_batch=True

export WANDB_MODE=online
export WANDB_DIR=${WANDB_DIR_CUSTOM}
export WANDB_CACHE_DIR=${WANDB_DIR_CUSTOM}
export WANDB_CONFIG_DIR=${WANDB_DIR_CUSTOM}
export WANDB_DATA_DIR=${WANDB_DIR_CUSTOM}
export WANDB_ARTIFACT_DIR=${WANDB_DIR_CUSTOM}
export WANDB_ARTIFACT_LOCATION=${WANDB_DIR_CUSTOM}
mkdir -p "${WANDB_DIR_CUSTOM}"
${PYTHON_BIN} -m recipe.attnrl.main_attnrl \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=question \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.enforce_eager=${free_cache_engine} \
    actor_rollout_ref.rollout.free_cache_engine=${free_cache_engine} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    actor_rollout_ref.rollout.max_num_seqs=${max_num_seqs} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.disable_log_stats=${disable_log_stats} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.balance_val_batch=${balance_val_batch} \
    +actor_rollout_ref.rollout.balance_gen_batch=${balance_gen_batch} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    +algorithm.split_criterion="entropy" \
    +algorithm.num_traces=30 \
    +algorithm.off_policy=True \
    reward_model.reward_manager=rllm_skywork \
    reward_model.launch_reward_fn_async=True \
    +reward_model.reward_kwargs.reward_type=v2_math-verify \
    +reward_model.reward_kwargs.reward_timeout=5 \
    trainer.total_epochs=100 \
    trainer.total_training_steps=1160 \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.logger=['console','wandb'] \
    +trainer.wandb_offline=${wandb_offline} \
    trainer.log_val_generations=5 \
    trainer.rollout_data_dir="${ROLLOUT_DIR}" \
    +trainer.rollout_data_freq=20 \
    trainer.validation_data_dir="${VAL_DIR}" \
    +trainer.reward_data_dir="${REWARD_DIR}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.save_freq=10 \
    trainer.resume_mode=disable \
    trainer.val_before_train=True \
    trainer.test_freq=5 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep} 2>&1 | tee ${LOG_DIR}/$(date +%m%d-%H%M%S).log
