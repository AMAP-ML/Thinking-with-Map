#!/bin/bash
set -e

RANK="$1"
if [[ -z "$RANK" ]]; then
  echo "Usage: $0 <RANK>"
  exit 1
fi

cd "$(dirname "$0")"

serper_api_key=""
google_map_api_key=""
amap_api_key=""
warm_start_file="./api_cache.json"
cache_file="./api_cache_${RANK}.json"

export SERPER_API_KEY="$serper_api_key"
export GOOGLE_MAP_API_KEY="$google_map_api_key"
export AMAP_API_KEY="$amap_api_key"
export WARM_START_FILE="$warm_start_file"
export CACHE_FILE="$cache_file"
export WARM_CACHE_ON_START="1"
export USE_CACHE="1"
export ENABLE_CACHE_SYNC="1"


python -m uvicorn api_server_redis:app --host 0.0.0.0 --port 8001 --workers 4 --access-log
