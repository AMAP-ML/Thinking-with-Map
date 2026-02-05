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

N_GPUS_PER_NODE=8
N_NODES=8

export BASE_MODEL='path/to/Qwen3-VL-30B-A3B-Instruct'
export EXPERIMENT_NAME=qwen3vl30ba3b-instruct-thinking-with-map-v2-sample6k


TRAIN_FILES_LIST=(
    "/path/to/MAPBench_train_v2_sample6k.parquet"
)
TRAIN_FILES="["
for ((i = 0; i < ${#TRAIN_FILES_LIST[@]}; i++)); do
    TRAIN_FILES+="\"${TRAIN_FILES_LIST[i]}\""
    if (( i < ${#TRAIN_FILES_LIST[@]} - 1 )); then
        TRAIN_FILES+=","
    fi
done
TRAIN_FILES+="]"
echo "TRAIN_FILES: ${TRAIN_FILES}"
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

ulimit -n 65535

ip_address=$MASTER_ADDR

if [ "$RANK" -eq 0 ]; then
    # 启动头部节点
    ray start --head --node-ip-address=$MASTER_ADDR --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --num-gpus=$N_GPUS_PER_NODE --object-store-memory=120000000000
    echo "ray status"
    ray status --address="$MASTER_ADDR:6379"
    sleep 20

    ray job submit --address=$RAY_ADDRESS \
        --runtime-env=verl/trainer/runtime_env.yaml \
        -- \
        python3 -m verl.trainer.main_ppo \
        --config-path="$CONFIG_PATH" \
        --config-name='geoagent_multiturn_grpo' \
        custom_reward_function.path=$PROJECT_DIR/verl/utils/reward_score/localization.py \
        custom_reward_function.name=compute_distance_score \
        algorithm.adv_estimator=grpo \
        data.train_batch_size=64 \
        data.val_batch_size=64 \
        data.max_prompt_length=8192 \
        data.max_response_length=4096 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.return_raw_chat=True \
        actor_rollout_ref.model.path=$BASE_MODEL \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.max_model_len=15000 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
        actor_rollout_ref.rollout.name=sglang \
        actor_rollout_ref.rollout.mode=async \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.rollout.n=16 \
        actor_rollout_ref.rollout.multi_turn.enable=True \
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns=8 \
        actor_rollout_ref.rollout.multi_turn.max_user_turns=8 \
        actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable \
        actor_rollout_ref.rollout.multi_turn.format=hermes \
        actor_rollout_ref.rollout.val_kwargs.n=4 \
        actor_rollout_ref.rollout.val_kwargs.top_k=60 \
        actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
        actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=left \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        trainer.ray_wait_register_center_timeout=1600 \
        trainer.critic_warmup=0 \
        trainer.val_before_train=False \
        trainer.val_only=False \
        trainer.logger="['console', 'swanlab']" \
        trainer.project_name=$PROJECT_NAME \
        trainer.experiment_name=$EXPERIMENT_NAME \
        trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
        trainer.nnodes=$N_NODES \
        trainer.save_freq=40 \
        trainer.test_freq=40 \
        trainer.validation_data_dir=${PROJECT_DIR}/verl_dump/$EXPERIMENT_NAME \
        trainer.default_local_dir=${PROJECT_DIR}/verl_checkpoints/$EXPERIMENT_NAME \
        data.train_files=${TRAIN_FILES} \
        data.val_files=${VALID_FILES} \
        actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
        trainer.total_epochs=1 \
        2>&1 | tee logs/$EXPERIMENT_NAME.log
else
    # 等待头部节点启动
    sleep 70
    echo "RAY_ADDRESS ${RAY_ADDRESS}, start to connect"
    while true; do
        # 检查主节点的 Ray 服务是否仍然在运行
        if ! curl -s --connect-timeout 5 $RAY_ADDRESS; then
            echo "Ray head node port 6379 is not reachable. Wait for next try."
        else 
            echo "Ray head node part 6379 is reachable. Start to connect."
            break
        fi
        sleep 30
    done
    ray start --address=$RAY_ADDRESS --num-gpus=8 --object-store-memory=120000000000
    trap 'echo "Shutting down worker node..."; ray stop; exit 0' SIGTERM
    echo "Worker node is ready and waiting for tasks..."
    while true; do
        # 检查主节点的 Ray 服务是否仍然在运行
        if ! curl -s --connect-timeout 5 $RAY_ADDRESS; then
            echo "Ray head node port 6379 is not reachable. Exiting worker node..."
            exit 0
        fi
        sleep 1800
    done
fi

