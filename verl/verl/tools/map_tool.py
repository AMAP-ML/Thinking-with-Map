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
import os
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4
from PIL import Image

import ray
import ray.actor

from verl.tools.utils.api_utils import perform_api_call, QueryType
from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")

################################################
## DEBUG
import os, time, socket, subprocess
import requests

def _sh(cmd: str) -> str:
    try:
        return subprocess.getoutput(cmd)
    except Exception as e:
        return f"<subprocess failed: {e}>"

def check_local_tool_server_once(logger, base_url="http://127.0.0.1:8001", wait_seconds=60):
    """
    Run inside Ray worker process.
    - Wait for /healthz up to wait_seconds
    - Print listen state (ss/netstat) and curl result
    """
    start = time.time()
    ok = False
    last_err = None

    for _ in range(wait_seconds):
        try:
            r = requests.get(f"{base_url}/healthz", timeout=1)
            if r.status_code == 200:
                ok = True
                break
            last_err = f"status={r.status_code} body={r.text[:100]}"
        except Exception as e:
            last_err = repr(e)
        time.sleep(1)

    msg = []
    msg.append("[startup-check] ok={} waited_s={} last_err={}".format(ok, int(time.time() - start), last_err))
    msg.append("[startup-check] pid={} hostname={}".format(os.getpid(), socket.gethostname()))
    msg.append("[startup-check] env RAY_NODE_IP_ADDRESS={}".format(os.environ.get("RAY_NODE_IP_ADDRESS")))
    msg.append("[startup-check] env KUBERNETES_SERVICE_HOST={}".format(os.environ.get("KUBERNETES_SERVICE_HOST")))

    cmd_ss = "ss -lntp 2>/dev/null | grep ':8001' || true"
    cmd_netstat = "netstat -lntp 2>/dev/null | grep ':8001' || true"
    cmd_curl = "curl -sS -m 1 {}/healthz -v 2>&1 | head -n 40".format(base_url)

    msg.append("[startup-check] ss -lntp | grep 8001:\n{}".format(_sh(cmd_ss)))
    msg.append("[startup-check] netstat -lntp | grep 8001:\n{}".format(_sh(cmd_netstat)))
    msg.append("[startup-check] curl /healthz:\n{}".format(_sh(cmd_curl)))

    logger.warning("\n".join(msg))
################################################


# Adapted from verl/tools/sandbox_fusion_tools.py
class PoolMode(Enum):
    """Execution pool mode enumeration."""

    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """Ray actor for rate limiting using token bucket algorithm."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0  # For observability
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        """Acquire a token from the bucket."""
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        """Release a token back to the bucket."""
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        """Get current number of acquired tokens."""
        return self.current_count


class SearchExecutionWorker:
    """Worker for executing search operations with optional rate limiting."""

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        ########################
        ## DEBUG
        # B: worker 启动自检（每个 actor 一次）
        try:
            check_local_tool_server_once(logger, base_url="http://127.0.0.1:8001", wait_seconds=60)
        except Exception as e:
            logger.warning(f"[startup-check] failed to run check: {e}")
        ########################

        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        """Initialize singleton rate limiter."""
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        """Health check method."""
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        """Execute function with optional rate limiting."""
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return fn(*fn_args, **fn_kwargs)
                except Exception as e:
                    # TODO we should make this available to the tool caller
                    logger.warning(f"Error when executing search: {e}")
        else:
            return fn(*fn_args, **fn_kwargs)


def init_search_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode
):
    """Initialize search execution pool."""
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(SearchExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")


class AmapInputTipsTool(BaseTool):
    """Amap input tips tool for get search tips in Amap POI search using external retrieval services.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the search tool
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize SearchTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition

        Example tool_schema:
            {
                "type": "function",
                "function": {
                    "name": "amap_input_tips",
                    "description": "搜索中国地点，获取相关POI建议",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "想要搜索的任意关键词"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Worker and rate limiting configuration
        self.num_workers = config.get("num_workers", 120)
        self.rate_limit = config.get("rate_limit", 120)
        self.timeout = config.get("timeout", 10)

        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_search_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )

        # Retrieval service configuration
        self.url = config.get("url")
        assert self.url, "Configuration must include 'url'"
        if self.url == "":
            raise ValueError("url is not set")

        logger.info(f"Initialized AmapInputTipsTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The instance id of the tool.
            tool_creation_response: The response of the tool when creating the instance.
        """
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
        }
        return instance_id, ToolResponse()

    def execute_api_call(self, instance_id: str, query: str, url: str, timeout: int):
        """Execute AmapInputTipsTool using api.

        Args:
            instance_id: Tool instance ID
            query: input query
            url: URL
            timeout: Request timeout in seconds

        Returns:
            Tuple of (result_text, metadata)
        """
        result_text, _, metadata = perform_api_call(
            url=url,
            query=query,
            query_type=QueryType.AMAP_INPUT_TIPS,
            concurrent_semaphore=None,  # Ray handles concurrency control
            timeout=timeout,
        )
        logger.debug(f"API call result for instance {instance_id}: {result_text}")
        return result_text, metadata

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the search tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing query and optional timeout

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The response str of the tool.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        timeout = self.timeout
        query = parameters.get("query")

        if not query or not isinstance(query, str):
            error_msg = "Error: 'query' is missing, empty, or not a list in parameters."
            logger.error(f"[API call] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=json.dumps({"result": error_msg})), 0.0, {}

        # Execute search using Ray execution pool
        try:
            result_text, metadata = await self.execution_pool.execute.remote(
                self.execute_api_call, instance_id, query, self.url, timeout
            )

            # Store results in instance dictionary
            self._instance_dict[instance_id]["reward"].append(result_text.strip())

            # Convert metadata to metrics
            metrics = {
                "query_count": metadata.get("query_count", 0),
                "status": metadata.get("status", "unknown"),
                "total_results": metadata.get("total_results", 0),
                "api_request_error": metadata.get("api_request_error"),
            }

            return ToolResponse(text=result_text), 0.0, metrics

        except Exception as e:
            error_result = json.dumps({"result": f"API call execution failed: {e}"})
            logger.error(f"[API call] Execution failed: {e}")
            return ToolResponse(text=error_result), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class AmapStaticMapTool(AmapInputTipsTool):
    """Amap static map tool for get statical map using external AMap services.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the search tool
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize SearchTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition

        Example tool_schema:
        {
            "type": "function",
            "function": {
                "name": "amap_static_map",
                "description": "根据经纬度，返回一个静态地图，包含附近所有区域，以便近一步判断该地点。注意只能查询中国区域",
                "parameters" = {
                    'type': 'object',
                    'properties': {
                        'latitude': {
                            'type': 'number',
                            'description': '查询地点的纬度，用正/负值表示北/南纬，例如39.9042'
                        },
                        'longitude': {
                            'type': 'number',
                            'description': '查询地点的经度，用正/负值表示东/西经，例如116.4074'
                        },
                    },
                    'required': ['latitude', 'longitude'],
                }
            }
        }
        """
        super().__init__(config, tool_schema)

    def execute_api_call(self, instance_id: str, latitude: float, longitude: float, url: str, timeout: int):
        """Execute AmapInputTipsTool using api.

        Args:
            instance_id: Tool instance ID
            query: input query
            url: URL
            timeout: Request timeout in seconds

        Returns:
            Tuple of (result_text, metadata)
        """
        result_text, result_image, metadata = perform_api_call(
            url=url,
            latitude=latitude,
            longitude=longitude,
            query_type=QueryType.AMAP_STATIC_MAP,
            concurrent_semaphore=None,  # Ray handles concurrency control
            timeout=timeout,
        )
        # logger.debug(f"API call result for instance {instance_id}: {result_text}")
        if result_image and isinstance(result_image, Image.Image):
            return result_text, result_image, metadata
        else:
            return result_text, None, metadata

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the search tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing query and optional timeout

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The response str of the tool.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        timeout = self.timeout
        latitude = parameters.get("latitude")
        longitude = parameters.get("longitude")

        if not latitude or not isinstance(latitude, float)\
            or not longitude or not isinstance(longitude, float):
            error_msg = "Error: 'latitude/longitude' is missing, empty, or not a list in parameters."
            logger.error(f"[API call] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=json.dumps({"result": error_msg})), 0.0, {}

        # Execute search using Ray execution pool
        try:
            result_text, result_image, metadata = await self.execution_pool.execute.remote(
                self.execute_api_call, instance_id, latitude, longitude, self.url, timeout
            )

            if result_image:
                # Store results in instance dictionary
                # self._instance_dict[instance_id]["reward"].append(result_text.strip())

                # Convert metadata to metrics
                metrics = {
                    "query_count": metadata.get("query_count", 0),
                    "status": metadata.get("status", "unknown"),
                    "total_results": metadata.get("total_results", 0),
                    "api_request_error": metadata.get("api_request_error"),
                }

                result_text= f"The nearby area map of ({latitude}, {longitude})."

                return ToolResponse(image=[result_image], text=result_text), 0.0, metrics

            else:
                return ToolResponse(text=result_text), 0.0, metrics
            
        except Exception as e:
            error_result = json.dumps({"result": f"API call execution failed: {e}"})
            logger.error(f"[API call] Execution failed: {e}")
            return ToolResponse(text=error_result), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class AmapSatelliteMapTool(AmapStaticMapTool):
    """Amap static map tool for get statical map using external AMap services.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the search tool
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize SearchTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition

        Example tool_schema:
        {
            "type": "function",
            "function": {
                "name": "amap_satellite_map",
                "description": "根据经纬度，返回一个卫星地图，包含附近所有区域，以便近一步判断该地点。注意只能查询中国区域",
                "parameters" = {
                    'type': 'object',
                    'properties': {
                        'latitude': {
                            'type': 'number',
                            'description': '查询地点的纬度，用正/负值表示北/南纬，例如39.9042'
                        },
                        'longitude': {
                            'type': 'number',
                            'description': '查询地点的经度，用正/负值表示东/西经，例如116.4074'
                        },
                    },
                    'required': ['latitude', 'longitude'],
                }
            }
        }
        """
        super().__init__(config, tool_schema)

    def execute_api_call(self, instance_id: str, latitude: float, longitude: float, url: str, timeout: int):
        """Execute AmapSatelliteMapTool using api.

        Args:
            

        Returns:
            Tuple of (result_text, metadata)
        """
        result_text, result_image, metadata = perform_api_call(
            url=url,
            latitude=latitude,
            longitude=longitude,
            query_type=QueryType.AMAP_SATELLITE_MAP,
            concurrent_semaphore=None,  # Ray handles concurrency control
            timeout=timeout,
        )
        # logger.debug(f"API call result for instance {instance_id}: {result_text}")
        if result_image and isinstance(result_image, Image.Image):
            return result_text, result_image, metadata
        else:
            return result_text, None, metadata


class AmapKeywordSearchTool(AmapInputTipsTool):
    """Amap input tips tool for get search tips in Amap POI search using external retrieval services.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the search tool
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize SearchTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition

        Example tool_schema:
            {
                "type": "function",
                "function": {
                    "name": "amap_keyword_search",
                    "description": "根据关键词进行搜索，搜索中国的POI",
                    "parameters" = {
                        'type': 'object',
                        'properties': {
                            'query': {
                                'description': '城市/区具体名称，如`北京市海淀区`请描述为`海淀区`',
                                'type': 'string',
                            },
                        },
                        'required': ['query'],
                    }
                }
            }
        """
        super().__init__(config, tool_schema)

    def execute_api_call(self, instance_id: str, query: str, url: str, timeout: int):
        """Execute AmapInputTipsTool using api.

        Args:
            instance_id: Tool instance ID
            query: input query
            url: URL
            timeout: Request timeout in seconds

        Returns:
            Tuple of (result_text, metadata)
        """
        result_text, _, metadata = perform_api_call(
            url=url,
            query=query,
            query_type=QueryType.AMAP_KEYWORD_SEARCH,
            concurrent_semaphore=None,  # Ray handles concurrency control
            timeout=timeout,
        )
        logger.debug(f"API call result for instance {instance_id}: {result_text}")
        return result_text, metadata
        

class AmapPOIDetailTool(AmapInputTipsTool):
    """Amap input tips tool for get search tips in Amap POI search using external retrieval services.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the search tool
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize SearchTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition

        Example tool_schema:
            {
                "type": "function",
                "function": {
                    "name": "amap_poi_detail",
                    "description": "根据中国的POI ID获取详细位置信息",
                    "parameters" = {
                        'type': 'object',
                        'properties': {
                            'query': {
                                'description': '需要查询的POI ID',
                                'type': 'string',
                            },
                        },
                        'required': ['query'],
                    }
                }
            }
        """
        super().__init__(config, tool_schema)

    def execute_api_call(self, instance_id: str, query: str, url: str, timeout: int):
        """Execute AmapInputTipsTool using api.

        Args:
            instance_id: Tool instance ID
            query: input query
            url: URL
            timeout: Request timeout in seconds

        Returns:
            Tuple of (result_text, metadata)
        """
        result_text, _, metadata = perform_api_call(
            url=url,
            query=query,
            query_type=QueryType.AMAP_POI_DETAIL,
            concurrent_semaphore=None,  # Ray handles concurrency control
            timeout=timeout,
        )
        logger.debug(f"API call result for instance {instance_id}: {result_text}")
        return result_text, metadata
    

class GoogleSearchTool(AmapInputTipsTool):
    """Amap input tips tool for get search tips in Amap POI search using external retrieval services.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the search tool
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize SearchTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition

        Example tool_schema:
            {
                "type": "function",
                "function": {
                    "name": "google_search",
                    "description": "Search online using the language of the target content (Chinese for Chinese content, English for English content). Keep queries concise.",
                    "parameters" = {
                        'type': 'object',
                        'properties': {
                            'query': {
                                'type': 'string',
                            }
                        },
                        'required': ['query'],
                    }
                }
            }
        """
        super().__init__(config, tool_schema)

    def execute_amap_api(self, instance_id: str, query: str, url: str, timeout: int):
        """Execute AmapInputTipsTool using api.

        Args:
            instance_id: Tool instance ID
            query: input query
            url: URL
            timeout: Request timeout in seconds

        Returns:
            Tuple of (result_text, metadata)
        """
        result_text, _, metadata = perform_api_call(
            url=url,
            query=query,
            query_type=QueryType.GOOGLE_SEARCH,
            concurrent_semaphore=None,  # Ray handles concurrency control
            timeout=timeout,
        )
        logger.debug(f"API call result for instance {instance_id}: {result_text}")
        return result_text, metadata
    

class GoogleMapSearchTool(AmapInputTipsTool):
    """Amap input tips tool for get search tips in Amap POI search using external retrieval services.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the search tool
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize SearchTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition

        Example tool_schema:
            {
                "type": "function",
                "function": {
                    "name": "google_map_search",
                    "description": "Search location (for precise latitude and longtitude) in Google Map.",
                    "parameters" = {
                        'type': 'object',
                        'properties': {
                            'query': {
                                'type': 'string',
                            }
                        },
                        'required': ['query'],
                    }
                }
            }
        """
        super().__init__(config, tool_schema)

    def execute_api_call(self, instance_id: str, query: str, url: str, timeout: int):
        """Execute AmapInputTipsTool using api.

        Args:
            instance_id: Tool instance ID
            query: input query
            url: URL
            timeout: Request timeout in seconds

        Returns:
            Tuple of (result_text, metadata)
        """
        result_text, _, metadata = perform_api_call(
            url=url,
            query=query,
            query_type=QueryType.GOOGLEMAP_SEARCH,
            concurrent_semaphore=None,  # Ray handles concurrency control
            timeout=timeout,
        )
        logger.debug(f"API call result for instance {instance_id}: {result_text}")
        return result_text, metadata

class GoogleStaticMapTool(AmapStaticMapTool):
    """Google static map tool for get statical map using external AMap services.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the search tool
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize SearchTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition

        Example tool_schema:
        {
            "type": "function",
            "function": {
                "name": "google_static_map",
                "description": "Input the latitude and longitude, a static map is returned, encompassing all nearby areas for further determining the location. Note that only areas outside of China can be queried.",
                "parameters" = {
                    'type': 'object',
                    'properties': {
                        'latitude': {
                            'type': 'number',
                            'description': 'The latitude of the query location. Use positive/negative values to represent north/south latitude, for example 40.714728.'
                        },
                        'longitude': {
                            'type': 'number',
                            'description': 'The longitude of the query location. Use positive/negative values to represent east/west longitude, for example -73.998672.'
                        },
                    },
                    'required': ['latitude', 'longitude'],
                }
            }
        }
        """
        super().__init__(config, tool_schema)

    def execute_api_call(self, instance_id: str, latitude: float, longitude: float, url: str, timeout: int):
        """Execute AmapSatelliteMapTool using api.

        Args:
            

        Returns:
            Tuple of (result_text, metadata)
        """
        result_text, result_image, metadata = perform_api_call(
            url=url,
            latitude=latitude,
            longitude=longitude,
            query_type=QueryType.GOOGLE_STATIC_MAP,
            concurrent_semaphore=None,  # Ray handles concurrency control
            timeout=timeout,
        )
        # logger.debug(f"API call result for instance {instance_id}: {result_text}")
        if result_image and isinstance(result_image, Image.Image):
            return result_text, result_image, metadata
        else:
            return result_text, None, metadata
