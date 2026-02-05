import os
import requests
import json
import time
import pathlib
import argparse
import uvicorn
import threading
import atexit
import langid
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any, Union
import json5
import resource
import traceback
from PIL import Image
from io import BytesIO
from fastapi import Response, HTTPException, Request
from filelock import FileLock, Timeout

STATIC_MAP_SEM = threading.Semaphore(int(os.environ.get("STATIC_MAP_CONCURRENCY", "2")))
SAT_MAP_SEM = threading.Semaphore(int(os.environ.get("SAT_MAP_CONCURRENCY", "2")))
KEYWORD_SEARCH_SEM = threading.Semaphore(int(os.environ.get("KEYWORD_SEARCH_CONCURRENCY", "80")))
INPUT_TIPS_SEM = threading.Semaphore(int(os.environ.get("INPUT_TIPS_CONCURRENCY", "80")))
POI_DETAILS_SEM = threading.Semaphore(int(os.environ.get("POI_DETAILS_CONCURRENCY", "80")))

LOCK_PATH = "/tmp/tool_cache_sync.leader.lock"
SYNC_LEADER_LOCK = FileLock(LOCK_PATH)
def acquire_leader_lock():
    try:
        SYNC_LEADER_LOCK.acquire(timeout=0)  # 抢不到就放弃
        return True
    except Exception:
        return False

from amap_static_map_sync import capture_map_html
from amap_satellite_map_sync import capture_satellite_html

INFLIGHT = 0
INFLIGHT_LOCK = threading.Lock()
START_TS = time.time()

try:
    import redis
except ImportError:
    print("Redis is not installed. Please run: pip install redis")
    exit(1)

from fastapi import FastAPI
from pydantic import BaseModel


def json_loads(text):
    text = text.strip('\n')
    if text.startswith('```') and text.endswith('\n```'):
        text = '\n'.join(text.split('\n')[1:-1])
    try:
        return json.loads(text)
    except json.decoder.JSONDecodeError as json_err:
        try:
            return json5.loads(text)
        except ValueError:
            raise json_err


class FileCounter:
    def __init__(self, path: str, init_if_missing: bool = True):
        self.path = pathlib.Path(path)
        self.lock = FileLock(str(self.path) + ".lock")

    def incr(self, delta: int = 1) -> Optional[int]:
        with self.lock:
            # 文件不存在：不更新（或你也可以选择初始化为0再更新）
            if not self.path.exists():
                return None

            try:
                raw = self.path.read_text(encoding="utf-8").strip()
                current = int(raw)  # 这里如果不是合法整数会抛异常
            except Exception as e:
                # 读/解析失败：不更新、不覆盖
                print(f"[FileCounter] read/parse failed, skip update: {e}")
                return None
            new_value = current + delta
            try:
                self.path.write_text(str(new_value), encoding="utf-8")
            except Exception as e:
                # 写失败：不保证更新成功，直接返回 None
                print(f"[FileCounter] write failed, skip update: {e}")
                return None
            return new_value

    def get(self) -> Optional[int]:
        with self.lock:
            if not self.path.exists():
                return None
            try:
                raw = self.path.read_text(encoding="utf-8").strip()
                return int(raw)
            except Exception as e:
                print(f"[FileCounter] read/parse failed: {e}")
                return None


# --- Helper: Redis to JSON Persistence Manager ---
class RedisCacheManager:
    """
    Manages the persistence of Redis cache to a JSON file.
    It performs periodic background saves and a final save on exit.
    """

    def __init__(self, redis_client: redis.Redis, warm_start_file: str, cache_file: str, sync_interval_seconds: float = 3600.0):
        if not redis_client:
            raise ValueError("A valid Redis client must be provided.")
        
        self.redis_client = redis_client
        self.warm_start_file = pathlib.Path(warm_start_file)
        self.cache_file = pathlib.Path(cache_file)
        self.sync_interval = sync_interval_seconds
        self._stop_event = threading.Event()
        
        # Ensure cache directory exists
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

        self.sync_file_lock = FileLock(str(self.cache_file) + ".sync.lock")

        # Start background sync thread
        self._sync_thread = threading.Thread(target=self._periodic_sync, daemon=True)
        self._sync_thread.start()
        
        # Register a final sync on program exit
        atexit.register(self.stop_and_sync)
        print(f"RedisCacheManager initialized. Will sync to '{self.cache_file}' every {self.sync_interval} seconds.")

    def _periodic_sync(self):
        """Background thread worker for periodic synchronization."""
        while not self._stop_event.wait(self.sync_interval):
            print("Performing periodic Redis to JSON sync...")
            self.sync_to_json()

    def sync_to_json(self, batch_size: int = 5000, max_keys: int | None = None):
        try:
            self.sync_file_lock.acquire(timeout=0)
        except Timeout:
            print("[sync] already running, skip", flush=True)
            return

        try:
            print("Starting sync from Redis to JSON...", flush=True)
            start_time = time.time()
            temp_file = self.cache_file.with_suffix('.tmp')
            n = 0

            with open(temp_file, "w", encoding="utf-8") as f:
                f.write("{")  # 顶层对象
                first = True

                batch = []
                for key in self.redis_client.scan_iter("*", count=1000):
                    batch.append(key)
                    if len(batch) >= batch_size:
                        vals = self.redis_client.mget(batch)
                        for k, v in zip(batch, vals):
                            if v is None:
                                continue
                            if not first:
                                f.write(",")
                            # 这里不再按 api_name 分组，直接 key->value 写出去（最省内存）
                            f.write(json.dumps(k, ensure_ascii=False))
                            f.write(":")
                            f.write(json.dumps(v, ensure_ascii=False))
                            first = False
                            n += 1
                            if max_keys and n >= max_keys:
                                break
                        batch.clear()
                    if max_keys and n >= max_keys:
                        break

                if batch:
                    vals = self.redis_client.mget(batch)
                    for k, v in zip(batch, vals):
                        if v is None:
                            continue
                        if not first:
                            f.write(",")
                        f.write(json.dumps(k, ensure_ascii=False))
                        f.write(":")
                        f.write(json.dumps(v, ensure_ascii=False))
                        first = False
                        n += 1

                f.write("}")

            temp_file.replace(self.cache_file)
            print(f"Synced {n} entries to '{self.cache_file}' in {time.time()-start_time:.2f}s", flush=True)
        except Exception as e:
            print(f"Error during sync_to_json: {e}", flush=True)
        
        finally:
            try:
                self.sync_file_lock.release()
            except Exception:
                pass


    def load_from_json(self):
        if not self.warm_start_file.exists():
            print(f"Warm start file '{self.warm_start_file}' not found. Skipping cache warming.")
            return

        print(f"Warming up cache from '{self.warm_start_file}'...")
        start_time = time.time()
        try:
            with open(self.warm_start_file, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)

            pipe = self.redis_client.pipeline()
            n = 0

            # 通过抽样判断格式：grouped = 顶层 value 仍是 dict
            is_grouped = False
            if isinstance(data, dict) and data:
                any_v = next(iter(data.values()))
                is_grouped = isinstance(any_v, dict)

            if is_grouped:
                # grouped: {api_name: {query: value}}
                for api_name, qdict in data.items():
                    if not isinstance(qdict, dict):
                        continue
                    for q, v in qdict.items():
                        pipe.set(f"{api_name}:{q}", v)
                        n += 1
                        if n % 5000 == 0:
                            pipe.execute()
                            pipe = self.redis_client.pipeline()
            else:
                # flat: { "api:query": value }
                if not isinstance(data, dict):
                    raise ValueError("Unsupported cache json format (top-level is not a dict)")
                for k, v in data.items():
                    pipe.set(k, v)
                    n += 1
                    if n % 5000 == 0:
                        pipe.execute()
                        pipe = self.redis_client.pipeline()

            pipe.execute()
            print(f"Successfully loaded {n} keys into Redis in {time.time() - start_time:.2f}s")

        except Exception as e:
            print(f"Error during load_from_json: {e}", flush=True)


    def stop_and_sync(self):
        print("Stopping Redis cache manager and performing final sync...")
        self._stop_event.set()
        try:
            self.sync_to_json()
        finally:
            try:
                self._sync_thread.join(timeout=30)
            except Exception:
                pass


# --- Core Search Engine Logic ---

class APICallEngine:
    """
    Bing search tool using Brightdata API, with a high-performance Redis cache
    and JSON file persistence.
    """

    def __init__(
        self,
        amap_api_key: str,
        serper_api_key: str,
        google_map_api_key: str,
        redis_client: redis.Redis,
        max_results: int = 10,
        result_length: int = 1000,
        tool_retry_count: int = 3,
        counter_file_path = None,
        use_cache = False,
    ):
        """
        Initialize the search tool.
        
        Args:
            amap_api_key: AMAP API key.
            serper_api_key: SERPER API key.
            redis_client: An initialized Redis client instance.
            max_results: Maximum number of search results to return.
            result_length: Maximum length of each result snippet.
            tool_retry_count: Number of retries for a failed search.
        """
        # API configuration
        self._amap_api_key = amap_api_key
        self._serper_api_key = serper_api_key
        self._google_map_api_key = google_map_api_key
        self._max_results = max_results
        self._result_length = result_length
        self._tool_retry_count = tool_retry_count

        self.amap_input_tips_url = "https://restapi.amap.com/v3/assistant/inputtips?keywords={keywords}&key={key}"
        self.amap_keyword_search_url = "https://restapi.amap.com/v3/place/text?keywords={keywords}&page_size=15&key={key}"
        self.amap_poi_detail_url = "https://restapi.amap.com/v3/place/detail?id={id}&key={key}"
        self.google_map_search_url = "https://google.serper.dev/maps"
        self.google_search_url = "https://google.serper.dev/search"
        self.google_static_map_url = "https://maps.googleapis.com/maps/api/staticmap?center={center}&zoom=19&size=500x500&maptype=roadmap&key={key}"
        
        # Redis cache client
        self.use_cache = use_cache
        self.redis_client = redis_client
        if counter_file_path:
            self.miss_counter = FileCounter(counter_file_path)
        else:
            self.miss_counter = None


    @property
    def name(self) -> str:
        return "api_call"

    @property
    def trigger_tag(self) -> str:
        return "api_call"

    def _make_request(self, query: str, query_type: str, timeout: int):
        """Sends a request to the Brightdata API."""
        lang_code, _ = langid.classify(query)
        mkt, setLang = ("zh-CN", "zh") if lang_code == 'zh' else ("en-US", "en")

        if query_type == "amap_input_tips":
            target_url = self.amap_input_tips_url.format(keywords=query, key=self._amap_api_key)
            return requests.get(target_url, timeout=timeout)
        elif query_type == "amap_keyword_search":
            target_url = self.amap_keyword_search_url.format(keywords=query, key=self._amap_api_key)
            return requests.get(target_url, timeout=timeout)
        elif query_type == "amap_poi_detail":
            target_url = self.amap_poi_detail_url.format(id=query, key=self._amap_api_key)
            return requests.get(target_url, timeout=timeout)
        elif query_type == "amap_static_map":
            return capture_map_html(query=query)
        elif query_type == "amap_satellite_map":
            return capture_satellite_html(query=query)
        elif query_type == "google_map_search":
            ## serper api
            payload = {'q': query}
            headers = {"X-API-KEY": f"{self.serper_api_key}", "Content-Type": "application/json"}
            return requests.request("POST", self.google_map_search_url, json=payload, headers=headers, timeout=timeout)
        elif query_type == "google_search":
            ## serper api
            payload = {'q': query, "hl": mkt}
            headers = {"X-API-KEY": f"{self.serper_api_key}", "Content-Type": "application/json"}
            return requests.request("POST", self.google_search_url, json=payload, headers=headers, timeout=timeout)
        elif query_type == "google_static_map":
            ## google map api
            target_url = self.google_static_map_url.format(center=query, key=f"{self.google_map_api_key}")
            return requests.get(target_url, timeout=timeout, proxies=proxies).content
        else:
            raise NotImplementedError

    def execute(self, query: str, query_type: str, timeout: int = 30):
        """
        Executes an API search query, caching by API type.
        """

        if not query:
            return "Empty query provided."

        # Redis key: api_name:query
        redis_key = f"{query_type}:{query}"
        
        if self.use_cache:
            try:
                cached_result = self.redis_client.get(redis_key)
                if cached_result:
                    print(f"Cache hit for [{query_type}] query: '{query}'")
                    return cached_result
            except Exception as e:
                print(f"Warning: Redis cache access failed: {e}")

        print(f"Skip Cache (or Cache miss) for [{query_type}] query: '{query}'. Performing live search...")
        try:
            if self.miss_counter:
                self.miss_counter.incr(1)
            t = time.time()
            response = self._make_request(query, query_type, timeout)
            print(f"[ext] pid={os.getpid()} type={query_type} http_ms={(time.time()-t)*1000:.1f}", flush=True)
            # print(response.text)
            if query_type == "amap_static_map" or query_type == "amap_satellite_map" or query_type == "google_static_map":
                return response
            else:
                response.raise_for_status()
                data = response.json()
                result = self._extract_and_format_results(data, query_type)

                if result and result.strip():
                    try:
                        self.redis_client.set(redis_key, result, ex=30*24*60*60) # 30天
                    except Exception as e:
                        print(f"Warning: Failed to save cache to Redis: {e}")

                return result

        except Exception as e:
            print(f"API call failed for [{query_type}] query '{query}': {e}")
            return ""

    def _extract_and_format_results(self, data: Dict, query_type: str) -> str:
        """Extracts and formats search results from the API response."""

        if query_type == "google_search":
            search_results = data['organic']
            content = '```\n{}\n```'.format('\n\n'.join([
                f"[{i}]\"{doc['title']}\n{doc.get('snippet', '')}\"{doc.get('date', '')}"
                for i, doc in enumerate(search_results, 1)
            ]))
            return content

        elif query_type == "google_map_search":
            lines = []
            for i, doc in enumerate(data['places'], 1):
                line = (
                    f"[{i}] {doc.get('title', '')}\n"
                    f"Address: {doc.get('address', '')}\n"
                    f"latitude: {doc.get('latitude', '')}\n"
                    f"longitude: {doc.get('longitude', '')}\n"
                    f"Type: {', '.join(doc.get('types', []))}\n"
                    f"Description: {doc.get('description', '')}"
                )
                lines.append(line)
            content = '```\n' + '\n\n'.join(lines) + '\n```'
            return content
        
        elif query_type == "amap_input_tips":
            tips_str_list = []
            for tip in data.get("tips", []):
                tip_lines = []
                for key in ["id", "name", "address", "typecode", "city"]:
                    tip_lines.append(f"{key}: {tip.get(key, '')}")
                tips_str_list.append("\n ".join(tip_lines))
            return "\n\n".join(tips_str_list)

        elif query_type == "amap_keyword_search":
            pois_str_list = []
            for poi in data.get("pois", []):
                poi_lines = []
                for key in ["id", "name", "address", "typecode"]:
                    poi_lines.append(f"{key}: {poi.get(key, '')}")
                pois_str_list.append("\n ".join(poi_lines))
            return "\n\n".join(pois_str_list)

        elif query_type == "amap_poi_detail":
            poi = data["pois"][0]
            detail_lines = []
            for key in [
                "id", "name", "location", "address",
                "business_area", "cityname", "type", "alias"
            ]:
                detail_lines.append(f"{key}: {poi.get(key, '')}")
            return "\n\n".join(detail_lines)

    def _format_results(self, results: Dict) -> str:
        """Formats search results into a readable text block."""
        if not results.get("chunk_content"):
            return "No search results found."

        formatted = []
        for idx, snippet in enumerate(results["chunk_content"][:self._max_results], 1):
            snippet = snippet[:self._result_length]
            formatted.append(f"{idx}: {snippet}")
        
        return "\n".join(formatted)

    def execute_with_retry(self, query: str, query_type: str):
        """Executes a search query with a built-in retry mechanism."""
        for i in range(self._tool_retry_count):
            try:
                result = self.execute(query, query_type)
                if result:
                    return result
                print(f"Attempt {i+1}/{self._tool_retry_count}: api call {query_type} returned empty output for '{query}'. Retrying...")
            except Exception as e:
                print(f"Attempt {i+1}/{self._tool_retry_count}: api call {query_type} failed for '{query}'. Error: {e}. Retrying...")
            time.sleep(1) # Wait a bit before retrying
        
        print(f"All {self._tool_retry_count} retries failed for query: '{query}'")
        return f"Api call failed for query: {query}"

    def api_call(self, query: str, query_type: str):
        return self.execute_with_retry(query, query_type)


# --- FastAPI Setup ---
app = FastAPI(title="API Call Proxy Server")

class SearchRequest(BaseModel):
    query: str

# ---- 全局状态（每个 uvicorn worker 进程会各自拥有一份）----
engine = None
cache_manager = None
redis_client = None

@app.middleware("http")
async def log_all_requests(request: Request, call_next):
    global INFLIGHT
    with INFLIGHT_LOCK:
        INFLIGHT += 1
        cur = INFLIGHT
    t0 = time.time()
    try:
        resp = await call_next(request)
        return resp
    finally:
        dt = (time.time() - t0) * 1000
        with INFLIGHT_LOCK:
            INFLIGHT -= 1
        print(
            f"[srv] pid={os.getpid()} tid={threading.get_ident()} "
            f"inflight={cur} {request.method} {request.url.path} cost_ms={dt:.1f}",
            flush=True,
        )

# --- API Routes ---
@app.post("/google_search")
def api_call_endpoint(request: SearchRequest):
    results = engine.api_call(request.query, "google_search")
    return results

@app.post("/google_map_search")
def api_call_endpoint(request: SearchRequest):
    results = engine.api_call(request.query, "google_map_search")
    return results


def _try_acquire_or_429(sem: "threading.Semaphore", name: str):
    if not sem.acquire(blocking=False):
        raise HTTPException(status_code=429, detail=f"too many {name} requests")


@app.post("/amap_input_tips")
def amap_input_tips_endpoint(request: SearchRequest):
    _try_acquire_or_429(INPUT_TIPS_SEM, "amap_input_tips")
    try:
        return engine.api_call(request.query, "amap_input_tips")
    finally:
        INPUT_TIPS_SEM.release()


@app.post("/amap_keyword_search")
def amap_keyword_search_endpoint(request: SearchRequest):
    _try_acquire_or_429(KEYWORD_SEARCH_SEM, "amap_keyword_search")
    try:
        return engine.api_call(request.query, "amap_keyword_search")
    finally:
        KEYWORD_SEARCH_SEM.release()


@app.post("/amap_poi_detail")
def amap_poi_detail_endpoint(request: SearchRequest):
    _try_acquire_or_429(POI_DETAILS_SEM, "amap_poi_detail")
    try:
        return engine.api_call(request.query, "amap_poi_detail")
    finally:
        POI_DETAILS_SEM.release()


@app.post("/amap_static_map")
def amap_static_map_endpoint(request: SearchRequest):
    _try_acquire_or_429(STATIC_MAP_SEM, "amap_static_map")
    try:
        results = engine.api_call(request.query, "amap_static_map")
        if isinstance(results, (bytes, bytearray)):
            # 你的截图是 PNG（page.screenshot 默认 PNG）
            return Response(content=results, media_type="image/png")
        return results
    finally:
        STATIC_MAP_SEM.release()


@app.post("/amap_satellite_map")
def amap_satellite_map_endpoint(request: SearchRequest):
    _try_acquire_or_429(SAT_MAP_SEM, "amap_satellite_map")
    try:
        results = engine.api_call(request.query, "amap_satellite_map")
        if isinstance(results, (bytes, bytearray)):
            return Response(content=results, media_type="image/png")
        return results
    finally:
        SAT_MAP_SEM.release()

@app.post("/sync_cache")
def sync_cache_endpoint(background: bool = True):
    global cache_manager
    if cache_manager is None:
        raise HTTPException(status_code=503, detail="cache_manager disabled or not leader")

    def _run():
        cache_manager.sync_to_json()

    if background:
        threading.Thread(target=_run, daemon=True).start()
        return {"message": "sync started (background)"}
    else:
        cache_manager.sync_to_json()
        return {"message": "sync finished"}

@app.get("/diag")
def diag():
    # 1分钟 loadavg：反映 CPU 压力/可运行队列
    try:
        load1, load5, load15 = os.getloadavg()
    except Exception:
        load1 = load5 = load15 = None

    # CPU 亲和性（是否绑核成功）
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except Exception:
        affinity = None

    # 进程资源（ru_maxrss 单位：KB，Linux）
    ru = resource.getrusage(resource.RUSAGE_SELF)

    with INFLIGHT_LOCK:
        inflight = INFLIGHT

    return {
        "pid": os.getpid(),
        "uptime_s": round(time.time() - START_TS, 1),
        "inflight": inflight,
        "threads": threading.active_count(),
        "loadavg": [load1, load5, load15],
        "affinity_len": None if affinity is None else len(affinity),
        "affinity_head": None if affinity is None else affinity[:20],
        "ru_maxrss_kb": getattr(ru, "ru_maxrss", None),
        "cpu_time_s": {
            "user": getattr(ru, "ru_utime", None),
            "sys": getattr(ru, "ru_stime", None),
        },
    }

@app.get("/healthz")
def healthz():
    try:
        # 可选：检查 redis
        # redis_client.ping()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

def connect_to_redis_with_retry(host, port, retries=20, delay=3):
    """
    尝试连接 Redis，如果失败则重试。
    """
    for i in range(retries):
        try:
            redis_client = redis.Redis(
                host=host,
                port=port,
                decode_responses=True,
                socket_connect_timeout=0.2,  # 连接建立超时（秒）
                socket_timeout=0.5,          # 单次读写超时（秒）
                retry_on_timeout=True,       # 超时允许重试（redis-py 自带）
                health_check_interval=30,    # 定期 PING，发现断连就重建
            )
            redis_client.ping()
            print(f"Successfully connected to Redis at {host}:{port}")
            return redis_client
        except redis.exceptions.BusyLoadingError as e:
            # 专门捕获 LOADING 错误
            print(f"Attempt {i+1}/{retries}: Redis is loading data. Retrying in {delay} seconds... Error: {e}")
            time.sleep(delay)
        except redis.exceptions.ConnectionError as e:
            # 捕获其他连接错误
            print(f"Attempt {i+1}/{retries}: Could not connect to Redis. Retrying in {delay} seconds... Error: {e}")
            time.sleep(delay)
            
    print(f"FATAL: Could not connect to Redis at {host}:{port} after {retries} attempts.")
    exit(1)

def parse_args_from_env_or_cli():
    args = argparse.Namespace()
    args.amap_api_key = os.environ["AMAP_API_KEY"]
    args.serper_api_key = os.environ.get("SERPER_API_KEY", "")
    args.google_map_api_key = os.environ.get("GOOGLE_MAP_API_KEY", "")

    args.redis_host = os.environ.get("REDIS_HOST", "localhost")
    args.redis_port = int(os.environ.get("REDIS_PORT", "6397"))

    args.warm_start_file = os.environ.get("WARM_START_FILE", "api_cache_redis.json")
    args.cache_file = os.environ.get("CACHE_FILE", "/tmp/api_cache_redis_dump.json")
    args.cache_sync_interval = float(os.environ.get("CACHE_SYNC_INTERVAL", "10000"))
    args.enable_cache_sync = os.environ.get("ENABLE_CACHE_SYNC", "0") == "1"

    args.warm_cache_on_start = os.environ.get("WARM_CACHE_ON_START", "0") == "1"
    args.counter_file_path = os.environ.get("COUNTER_FILE_PATH")  # 可为空
    args.use_cache = os.environ.get("USE_CACHE", "0") == "1"

    args.tool_retry_count = int(os.environ.get("TOOL_RETRY_COUNT", "3"))
    args.max_results = int(os.environ.get("MAX_RESULTS", "10"))
    args.result_length = int(os.environ.get("RESULT_LENGTH", "1000"))
    return args


@app.on_event("startup")
def on_startup():
    global engine, cache_manager, redis_client

    args = parse_args_from_env_or_cli()

    # 1) redis
    redis_client = connect_to_redis_with_retry(args.redis_host, args.redis_port)

    # 2) cache manager（可选：训练时建议先关掉周期 sync，避免抖动）
    warm_start_file = str(pathlib.Path(args.warm_start_file).expanduser())
    cache_file = str(pathlib.Path(args.cache_file).expanduser())
    cache_manager = None
    if not args.enable_cache_sync:
        print(f"[startup] cache sync disabled by env pid={os.getpid()}", flush=True)
    else:
        if acquire_leader_lock():
            print(f"[startup] cache sync leader acquired pid={os.getpid()}", flush=True)
            cache_manager = RedisCacheManager(
                redis_client=redis_client,
                warm_start_file=warm_start_file,
                cache_file=cache_file,
                sync_interval_seconds=args.cache_sync_interval,
            )
            if args.warm_cache_on_start:
                cache_manager.load_from_json()
        else:
            print(f"[startup] cache sync leader NOT acquired pid={os.getpid()}", flush=True)

    # 3) engine
    engine = APICallEngine(
        amap_api_key=args.amap_api_key,
        redis_client=redis_client,
        tool_retry_count=args.tool_retry_count,
        max_results=args.max_results,
        result_length=args.result_length,
        counter_file_path=args.counter_file_path,
        use_cache=args.use_cache,
    )

    print(f"[startup] engine ready pid={os.getpid()} use_cache={args.use_cache}", flush=True)


@app.on_event("shutdown")
def on_shutdown():
    global cache_manager
    try:
        if cache_manager:
            cache_manager.stop_and_sync()
    except Exception as e:
        print(f"[shutdown] cache_manager stop failed: {e}", flush=True)
    try:
        if SYNC_LEADER_LOCK.is_locked:
            SYNC_LEADER_LOCK.release()
    except Exception:
        pass

def _heartbeat():
    while True:
        try:
            with INFLIGHT_LOCK:
                inflight = INFLIGHT
            try:
                la = os.getloadavg()
            except Exception:
                la = (None, None, None)
            print(f"[hb] pid={os.getpid()} inflight={inflight} load1={la[0]}", flush=True)
        except Exception as e:
            print(f"[hb] err: {e}", flush=True)
        time.sleep(5)

# @app.on_event("startup")
# def start_hb():
#     t = threading.Thread(target=_heartbeat, daemon=True)
#     t.start()

# --- Main Application Entry Point ---
if __name__ == "__main__":
    # 仍然允许命令行启动，但推荐你在集群里用环境变量 + uvicorn --workers
    uvicorn.run("tool_server:app", host="0.0.0.0", port=8001, workers=1, access_log=True)