# amap_static_map_sync.py
import atexit
import threading
import time
import urllib.parse
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Optional

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

CHROME_PATH = "./chrome-linux/chrome"

HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Map</title>
  <style>
    html, body, #container {{ width: 100%; height: 100%; margin: 0; padding: 0; }}
  </style>
</head>
<body>
<div id="container"></div>

<script>
  window.__MAP_READY__ = false;
  window.__MAP_ERROR__ = null;
</script>

<script src="https://webapi.amap.com/maps?v=2.0&key={api_key}"></script>

<script>
(function () {{
  try {{
    if (!window.AMap) throw new Error("AMap not loaded (key/network?)");

    var map = new AMap.Map('container', {{
      viewMode: '2D',
      zoom: {zoom},
      center: [{query}]
    }});

    map.on('tilesloaded', function () {{
      window.__MAP_READY__ = true;
    }});

    // 兜底，避免 tilesloaded 不触发导致无限等待
    setTimeout(function () {{
      window.__MAP_READY__ = true;
    }}, 2500);

  }} catch (e) {{
    window.__MAP_ERROR__ = String(e && e.message ? e.message : e);
  }}
}})();
</script>
</body>
</html>
"""

# ---------------- Playwright single-thread worker (per process) ----------------

@dataclass
class _Job:
    query: str
    api_key: str
    zoom: int
    width: int
    height: int
    hard_timeout_ms: int
    done: threading.Event
    result: Optional[bytes] = None


# 队列不要太大：队列大只会让请求排队变长，训练端更容易 30s 超时
_JOB_QUEUE_MAXSIZE = 64
_job_q: "Queue[_Job]" = Queue(maxsize=_JOB_QUEUE_MAXSIZE)

_worker_t: Optional[threading.Thread] = None
_worker_lock = threading.Lock()
_stop = threading.Event()

# 定期重启 browser，防止长跑内存泄漏/变慢
_RESTART_EVERY_N = 200
_shot_n = 0


def _launch_browser():
    pw = sync_playwright().start()
    browser = pw.chromium.launch(
        headless=True,
        executable_path=CHROME_PATH,
        args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
        ],
    )
    return pw, browser


def _worker_loop():
    global _shot_n
    pw, browser = _launch_browser()

    def restart():
        nonlocal pw, browser
        try:
            browser.close()
        except Exception:
            pass
        try:
            pw.stop()
        except Exception:
            pass
        pw, browser = _launch_browser()

    while not _stop.is_set():
        try:
            job = _job_q.get(timeout=0.2)
        except Empty:
            continue

        try:
            t0 = time.time()
            html_str = HTML_TEMPLATE.format(api_key=job.api_key, zoom=job.zoom, query=job.query)
            data_url = "data:text/html;charset=utf-8," + urllib.parse.quote(html_str)

            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
                device_scale_factor=1,
                viewport={"width": job.width, "height": job.height},
            )
            page = context.new_page()

            # 注意：不要在高并发下大量 print console，会拖慢；需要时再打开
            # page.on("console", lambda msg: print("[page console]", msg.type, msg.text, flush=True))
            # page.on("pageerror", lambda err: print("[page error]", err, flush=True))
            # page.on("requestfailed", lambda req: print("[request failed]", req.url, req.failure, flush=True))

            page.goto(data_url, wait_until="domcontentloaded", timeout=min(8000, job.hard_timeout_ms))
            page.wait_for_selector("#container", timeout=min(3000, job.hard_timeout_ms))

            remain = max(1000, job.hard_timeout_ms - int((time.time() - t0) * 1000))
            page.wait_for_function(
                "window.__MAP_READY__ === true || window.__MAP_ERROR__ !== null",
                timeout=remain,
            )

            err = page.evaluate("window.__MAP_ERROR__")
            if err:
                job.result = None
            else:
                remain = max(1000, job.hard_timeout_ms - int((time.time() - t0) * 1000))
                job.result = page.screenshot(timeout=remain)  # PNG bytes

        except PWTimeout:
            job.result = None
        except Exception:
            job.result = None
        finally:
            try:
                page.close()
            except Exception:
                pass
            try:
                context.close()
            except Exception:
                pass

            _shot_n += 1
            if _RESTART_EVERY_N and (_shot_n % _RESTART_EVERY_N == 0):
                restart()

            job.done.set()
            _job_q.task_done()

    # cleanup
    try:
        browser.close()
    except Exception:
        pass
    try:
        pw.stop()
    except Exception:
        pass


def _ensure_worker():
    global _worker_t
    with _worker_lock:
        if _worker_t is None or not _worker_t.is_alive():
            _stop.clear()
            _worker_t = threading.Thread(
                target=_worker_loop,
                name="amap-playwright-worker",
                daemon=True,
            )
            _worker_t.start()


def capture_map_html(
    query: str,
    api_key: str,
    zoom: int = 19,
    width: int = 500,
    height: int = 500,
    hard_timeout_ms: int = 10000,
) -> Optional[bytes]:
    """
    线程安全：可被 FastAPI/uvicorn 并发调用。
    Playwright 只在专用线程内执行。
    """
    _ensure_worker()

    job = _Job(
        query=query,
        api_key=api_key,
        zoom=zoom,
        width=width,
        height=height,
        hard_timeout_ms=hard_timeout_ms,
        done=threading.Event(),
    )

    # 队列满直接快速失败：避免请求堆积导致 client 30s timeout
    try:
        _job_q.put(job, timeout=0.02)
    except Exception:
        return None

    # 等待结果（额外 +2s 给线程调度/收尾）
    wait_s = hard_timeout_ms / 1000.0 + 2.0
    if not job.done.wait(timeout=wait_s):
        return None
    return job.result


def _cleanup():
    _stop.set()


atexit.register(_cleanup)


if __name__ == "__main__":
    from PIL import Image
    from io import BytesIO

    img_bytes = capture_map_html("111.286131,30.693383", api_key="", zoom=18)
    if img_bytes:
        Image.open(BytesIO(img_bytes)).save("map.png")
        print("saved map.png")
    else:
        print("failed")
