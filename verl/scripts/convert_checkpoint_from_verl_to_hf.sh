python ./scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --hf_model_path ./models/Qwen3-VL-30B-A3B-Instruct \
    --local_dir verl/verl_checkpoints/geoagent-qwen3vl30ba3b-instruct-amapv31sample6k-n16turn8-pass4-grpo-8node/global_step_92/actor \
    --target_dir verl/verl_checkpoints/merge/amapv31sample6k_qwen3vl30ba3b_grpo_step92