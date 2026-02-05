import json

from .schemas import OpenAIFunctionToolSchema, ToolResponse
from .utils.api_utils import perform_api_call, QueryType


def execute_api_call(instance_id: str, latitude: float, longitude: float, url: str, timeout: int):
    """Execute AmapInputTipsTool using api.

    Args:
        instance_id: Tool instance ID
        query: input query
        url: URL
        timeout: Request timeout in seconds

    Returns:
        Tuple of (result_text, metadata)
    """
    _, result_image, metadata = perform_api_call(
        url=url,
        latitude=latitude,
        longitude=longitude,
        query_type=QueryType.AMAP_STATIC_MAP,
        concurrent_semaphore=None,  # Ray handles concurrency control
        timeout=timeout,
    )
    # logger.debug(f"API call result for instance {instance_id}: {result_text}")
    return result_image, metadata

async def execute(instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
    """Execute the search tool.

    Args:
        instance_id: The instance ID of the tool
        parameters: Tool parameters containing query and optional timeout

    Returns: tool_response, tool_reward_score, tool_metrics
        tool_response: The response str of the tool.
        tool_reward_score: The step reward score of the tool.
        tool_metrics: The metrics of the tool.
    """
    timeout = 3000
    latitude = parameters.get("latitude")
    longitude = parameters.get("longitude")

    if not latitude or not isinstance(latitude, float)\
        or not longitude or not isinstance(longitude, float):
        error_msg = "Error: 'latitude/longitude' is missing, empty, or not a list in parameters."
        return ToolResponse(text=json.dumps({"result": error_msg})), 0.0, {}

    # Execute search using Ray execution pool
    try:
        result_image, metadata = await self.execution_pool.execute.remote(
            self.execute_api_call, instance_id, latitude, longitude, self.url, timeout
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

        result_text= f"The nearby area map of ({latitude}, {longitude})."

        return ToolResponse(image=[result_image], text=result_text), 0.0, metrics
        
    except Exception as e:
        error_result = json.dumps({"result": f"API call execution failed: {e}"})
        logger.error(f"[API call] Execution failed: {e}")
        return ToolResponse(text=error_result), 0.0, {"error": str(e)}

async def calc_reward(self, instance_id: str, **kwargs) -> str:
    return self._instance_dict[instance_id]["reward"]

async def release(self, instance_id: str, **kwargs) -> None:
    if instance_id in self._instance_dict:
        del self._instance_dict[instance_id]