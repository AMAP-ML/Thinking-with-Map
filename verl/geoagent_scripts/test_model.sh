set -x

ulimit -n 65535

export RAY_PLASMA_STORE_MEMORY=$((256*1024*1024*1024))
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export SWANLAB_API_KEY="xxxxxx"
export SWANLAB_MODE="cloud"

PROJECT_NAME="Thinking-with-Map"


PROJECT_DIR="path/to/project/verl"
cd $PROJECT_DIR

CONFIG_PATH="$PROJECT_DIR/geoagent_scripts/config"
TOOL_CONFIG="$CONFIG_PATH/geoagent_amap_tool_config.yaml"


n_gpus_per_node=8

export BASE_MODEL='/mnt/workspace/common/models/Qwen3-VL-30B-A3B-Instruct'
export EXPERIMENT_NAME=basemodel-qwen3vl30b-test-datav1

TRAIN_FILES="/path/to/MAPBench_train_v2_sample6k.parquet"
VALID_FILES_LIST=(
    "/path/to/MAPBench_test_v2.parquet"
    "/path/to/MAPBench_test_v1_easy.parquet"
    "/path/to/MAPBench_test_v1_hard.parquet"
)
VALID_FILES="["
for ((i = 0; i < ${#VALID_FILES_LIST[@]}; i++)); do
    VALID_FILES+="\"${VALID_FILES_LIST[i]}\""
    if (( i < ${#VALID_FILES_LIST[@]} - 1 )); then
        VALID_FILES+=","
    fi
done
VALID_FILES+="]"
echo "VALID_FILES: ${VALID_FILES}"


# ray job submit --address=$RAY_DASHBOARD_ADDRESS \
#     --runtime-env=verl/trainer/runtime_env.yaml \
#     -- \
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='geoagent_grpo' \
    custom_reward_function.path=$PROJECT_DIR/verl/utils/reward_score/localization.py \
    custom_reward_function.name=compute_distance_score \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=512 \
    data.val_batch_size=64 \
    data.max_prompt_length=8192 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.max_model_len=15000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.logger="['console', 'swanlab']" \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VALID_FILES} \
    trainer.total_epochs=1 \
    2>&1 | tee logs/$EXPERIMENT_NAME.log

