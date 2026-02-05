## V2
python preprocess_thinking_with_map.py \
    --input_jsonl /path/to/MAPBench/train_metadata_v2.jsonl \
    --image_dir /path/to/MAPBench \
    --output_path /path/to/MAPBench/MAPBench_train_v2.parquet \
    --split train \
    --data_source MAPBench_v2

## V2 sample 6k
python preprocess_thinking_with_map.py \
    --input_jsonl /path/to/MAPBench/train_metadata_v2_sample6k.jsonl \
    --image_dir /path/to/MAPBench \
    --output_path /path/to/MAPBench/MAPBench_train_v2_sample6k.parquet \
    --split train \
    --data_source MAPBench_v2

## V2 test
python preprocess_thinking_with_map.py \
    --input_jsonl /path/to/MAPBench/test_metadata_v2.jsonl \
    --image_dir /path/to/MAPBench \
    --output_path /path/to/MAPBench/MAPBench_test_v2.parquet \
    --split test \
    --data_source MAPBench_v2

## V1 train
python preprocess_thinking_with_map.py \
    --input_jsonl /path/to/MAPBench/test_metadata_v1.jsonl \
    --image_dir /path/to/MAPBench \
    --output_path /path/to/MAPBench/MAPBench_train_v1.parquet \
    --split train \
    --data_source MAPBench_v1

## V1 test easy
python preprocess_thinking_with_map.py \
    --input_jsonl /path/to/MAPBench/test_metadata_v1_easy.jsonl \
    --image_dir /path/to/MAPBench \
    --output_path /path/to/MAPBench/MAPBench_test_v1_easy.parquet \
    --split test \
    --data_source MAPBench_v1_easy

## V1 test hard
python preprocess_thinking_with_map.py \
    --input_jsonl /path/to/MAPBench/test_metadata_v1_hard.jsonl \
    --image_dir /path/to/MAPBench \
    --output_path /path/to/MAPBench/MAPBench_test_v1_hard.parquet \
    --split test \
    --data_source MAPBench_v1_hard

