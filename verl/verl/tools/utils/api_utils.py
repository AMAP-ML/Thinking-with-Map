# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import threading
import time
import traceback
import uuid
from typing import Any, Optional
from PIL import Image
from io import BytesIO
from PIL import UnidentifiedImageError

import requests
from enum import Enum

class QueryType(str, Enum):
    AMAP_INPUT_TIPS = "amap_input_tips"
    AMAP_KEYWORD_SEARCH = "amap_keyword_search"
    AMAP_POI_DETAIL = "amap_poi_detail"
    AMAP_STATIC_MAP = "amap_static_map"
    AMAP_SATELLITE_MAP = "amap_satellite_map"
    GOOGLEMAP_SEARCH = "google_map_search"
    GOOGLE_SEARCH = "google_search"
    GOOGLE_STATIC_MAP = "google_static_map"


DEFAULT_TIMEOUT = 10  # Default search request timeout
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 1
API_TIMEOUT = 10

logger = logging.getLogger(__name__)


#####################################
#### DEBUG
import os, time, socket, subprocess, threading
from urllib.parse import urlparse

_DIAG_ONCE = False
_DIAG_LOCK = threading.Lock()

def _sh(cmd: str) -> str:
    try:
        return subprocess.getoutput(cmd)
    except Exception as e:
        return f"<subprocess failed: {e}>"

def diag_localhost_8001(base_url: str, curl_timeout: float = 1.0) -> str:
    lines = []
    lines.append(f"[tool-diag] time={time.strftime('%F %T')}")
    lines.append(f"[tool-diag] pid={os.getpid()} hostname={socket.gethostname()}")
    lines.append(f"[tool-diag] env HOSTNAME={os.environ.get('HOSTNAME')}")
    lines.append(f"[tool-diag] env KUBERNETES_SERVICE_HOST={os.environ.get('KUBERNETES_SERVICE_HOST')}")
    lines.append(f"[tool-diag] env RAY_NODE_IP_ADDRESS={os.environ.get('RAY_NODE_IP_ADDRESS')}")
    lines.append(f"[tool-diag] env HTTP_PROXY={os.environ.get('HTTP_PROXY')}")
    lines.append(f"[tool-diag] env HTTPS_PROXY={os.environ.get('HTTPS_PROXY')}")
    lines.append(f"[tool-diag] env NO_PROXY={os.environ.get('NO_PROXY')}")

    lines.append("[tool-diag] ip addr:\n" + _sh("ip addr | head -n 60"))
    lines.append("[tool-diag] ss -lntp | grep 8001:\n" + _sh("ss -lntp 2>/dev/null | grep ':8001' || true"))
    lines.append("[tool-diag] netstat -lntp | grep 8001:\n" + _sh("netstat -lntp 2>/dev/null | grep ':8001' || true"))
    lines.append("[tool-diag] curl healthz:\n" + _sh(f"curl -sS -m {curl_timeout} {base_url}/healthz -v 2>&1 | head -n 80"))
    lines.append("[tool-diag] curl diag:\n" + _sh(f"curl -sS -m {curl_timeout + 1} {base_url}/diag -v 2>&1 | head -n 120"))
    
    lines.append("[tool-diag] ss -lntp | grep 6397:\n" + _sh("ss -lntp 2>/dev/null | grep ':6397' || true"))
    lines.append("[tool-diag] ps redis:\n" + _sh("ps -ef | grep redis-server | grep -v grep | head"))
    lines.append("[tool-diag] redis-cli ping:\n" + _sh("redis-cli -h 127.0.0.1 -p 6397 PING 2>&1 | head"))

    return "\n".join(lines)

def maybe_log_diag(logger, base_url: str):
    global _DIAG_ONCE
    with _DIAG_LOCK:
        if _DIAG_ONCE:
            return
        _DIAG_ONCE = True
    try:
        logger.warning(diag_localhost_8001(base_url=base_url))
    except Exception:
        pass
#####################################


def call_api(
    url: str,
    query: str,
    latitude: float,
    longitude: float,
    query_type: QueryType,
    return_scores: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[str, Optional[Image.Image], Optional[str]]:
    """
    Calls the specific API to perform retrieval with retry logic for various errors,
    using increasing delay between retries. Logs internal calls with a unique ID.

    Args:
        url: The URL of the API.
        query: query.
        query_type: API type.
        return_scores: Whether to return scores.
        timeout: Request timeout in seconds.

    Returns:
        A tuple (response_json, image, error_message).
        If successful, response_json is the API's returned JSON object, image is the API's returned Image.Image obejtc (optional), error_message is None.
        If failed after retries, response_json is None, error_message contains the error information.
    """
    request_id = str(uuid.uuid4())
    log_prefix = f"[QueryType: {query_type} API call Request ID: {request_id}, query={query}&lat={latitude}&lon={longitude}] "

    if query_type == QueryType.AMAP_INPUT_TIPS:
        payload = {
            "query": query,
        }
    elif query_type == QueryType.AMAP_KEYWORD_SEARCH:
        payload = {
            "query": query,
        }
    elif query_type == QueryType.AMAP_POI_DETAIL:
        payload = {
            "query": query,
        }
    elif query_type == QueryType.AMAP_STATIC_MAP:
        payload = {
            "query": f"{longitude},{latitude}"
        }
    elif query_type == QueryType.AMAP_SATELLITE_MAP:
        payload = {
            "query": f"{longitude},{latitude}"
        }
    elif query_type == QueryType.GOOGLE_SEARCH:
        payload = {
            "query": query,
        }
    elif query_type == QueryType.GOOGLEMAP_SEARCH:
        payload = {
            "query": query,
        }
    elif query_type == QueryType.GOOGLE_STATIC_MAP:
        payload = {
            "query": f"{latitude},{longitude}",
        }
    else:
        raise NotImplementedError

    headers = {"Content-Type": "application/json"}

    last_error = None

    parsed = urlparse(url)
    base_url = f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(
                f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: Calling {query_type} API at {url}"
            )
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            # Check for Gateway Timeout (504) and other server errors for retrying
            if response.status_code in [500, 502, 503, 504]:
                last_error = (
                    f"{log_prefix}API Request Error: Server Error ({response.status_code}) on attempt "
                    f"{attempt + 1}/{MAX_RETRIES}"
                )
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                    time.sleep(delay)
                continue

            # Check for other HTTP errors (e.g., 4xx)
            response.raise_for_status()

            # breakpoint()

            # If successful (status code 2xx)
            logger.info(f"{log_prefix} API call successful on attempt {attempt + 1}")

            try:
                if query_type in (QueryType.AMAP_STATIC_MAP, QueryType.AMAP_SATELLITE_MAP, QueryType.GOOGLE_STATIC_MAP):
                    img = Image.open(BytesIO(response.content))
                    img.load()   # 验证图片完整性（只检查头部）
                    return None, img, None
                else:
                    return response.text.replace('\\n', '\n'), None, None
            except UnidentifiedImageError as e:
                last_error = f"{log_prefix}Invalid image data: {e}"
                break

        except requests.exceptions.ConnectionError as e:
            last_error = f"{log_prefix}Connection Error: {e}"
            logger.warning(last_error)
            ########
            ## DEBUG
            # maybe_log_diag(logger, base_url)
            ########
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.Timeout as e:
            last_error = f"{log_prefix}Timeout Error: {e}"
            logger.warning(last_error)
            ########
            ## DEBUG
            # maybe_log_diag(logger, base_url)
            ########
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"
            break  # Exit retry loop on other request errors
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}, Response: {raw_response_text[:200]}"
            break  # Exit retry loop on JSON decode errors
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"
            break  # Exit retry loop on other unexpected errors

    # If loop finishes without returning success, return the last recorded error
    logger.error(f"{log_prefix} API call failed. Last error: {last_error}")
    return None, None, last_error.replace(log_prefix, "API Call Failed: ") if last_error else "API Call Failed after retries"


def perform_api_call(
    url: str,
    query_type: QueryType,
    query: str = None,
    latitude: float = None,
    longitude: float = None,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[str, Image.Image, dict[str, Any]]:
    """
    Performs a single batch search for multiple queries (original search tool behavior).

    Args:
        url: The URL of API.
        query: query.
        latitude: lat.
        longitude: lon.
        query_type: 
        concurrent_semaphore: Optional semaphore for concurrency control.
        timeout: Request timeout in seconds.

    Returns:
        A tuple (result_text, metadata).
        result_text: The result JSON string.
        metadata: Metadata dictionary for the batch search.
    """
    logger.info(f"Starting api call for {query_type}: {query}")

    api_response = None
    error_msg = None

    try:
        if concurrent_semaphore:
            with concurrent_semaphore:
                api_response, image, error_msg = call_api(
                    url=url,
                    query=query,
                    latitude=latitude,
                    longitude=longitude,
                    query_type=query_type,
                    return_scores=True,
                    timeout=timeout,
                )
        else:
            api_response, image, error_msg = call_api(
                url=url,
                query=query,
                latitude=latitude,
                longitude=longitude,
                query_type=query_type,
                return_scores=True,
                timeout=timeout,
            )
    except Exception as e:
        error_msg = f"API Request Exception during api call: {e}"
        logger.error(f"API Call: {error_msg}")
        traceback.print_exc()

    metadata = {
        "query_type": query_type,
        "query": query,
        "latitude": latitude,
        "longitude": longitude,
        "api_request_error": error_msg,
        "api_response": None,
        "status": "unknown",
        "total_results": 0,
        "formatted_result": None,
    }

    result_text = json.dumps({"result": "API call failed or timed out after retries."}, ensure_ascii=False)

    if error_msg:
        metadata["status"] = "api_error"
        result_text = json.dumps(obj={"result": f"{query_type} API call error: {error_msg}"}, ensure_ascii=False)
        logger.error(f"{query_type}: API error occurred: {error_msg}")
    elif api_response or (image and isinstance(image, Image.Image)):
        logger.debug(f"{query_type}: API Response: {api_response}")
        metadata["api_response"] = api_response

        try:       
            result_text = api_response
            metadata["status"] = "success"
            metadata["total_results"] = 1
            metadata["formatted_result"] = result_text

        except Exception as e:
            error_msg = f"Error processing {query_type} api call results: {e}"
            result_text = json.dumps({"result": error_msg}, ensure_ascii=False)
            metadata["status"] = "processing_error"
            logger.error(f"{query_type} api call: {error_msg}")
    else:
        metadata["status"] = "unknown_api_state"
        result_text = json.dumps(
            {"result": "Unknown API state (no response and no error message)."}, ensure_ascii=False
        )
        logger.error(f"{query_type} api call: Unknown API state.")

    return result_text, image, metadata
