# amap_satellite_map_sync.py
import atexit
import threading
import time
import urllib.parse
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Optional

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

CHROME_PATH = "/path/to/chrome-linux/chrome"

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
      center: [{query}],
      layers: [
        new AMap.TileLayer.Satellite(),
        new AMap.TileLayer.RoadNet()
      ]
    }});

    map.on('tilesloaded', function () {{
      window.__MAP_READY__ = true;
    }});

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


_JOB_QUEUE_MAXSIZE = 64
_job_q: "Queue[_Job]" = Queue(maxsize=_JOB_QUEUE_MAXSIZE)

_worker_t: Optional[threading.Thread] = None
_worker_lock = threading.Lock()
_stop = threading.Event()

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

        context = None
        page = None
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
                job.result = page.screenshot(timeout=remain)

        except PWTimeout:
            job.result = None
        except Exception:
            job.result = None
        finally:
            try:
                if page is not None:
                    page.close()
            except Exception:
                pass
            try:
                if context is not None:
                    context.close()
            except Exception:
                pass

            _shot_n += 1
            if _RESTART_EVERY_N and (_shot_n % _RESTART_EVERY_N == 0):
                restart()

            job.done.set()
            _job_q.task_done()

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
                name="amap-satellite-playwright-worker",
                daemon=True,
            )
            _worker_t.start()


def capture_satellite_html(
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

    try:
        _job_q.put(job, timeout=0.02)
    except Exception:
        return None

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

    img_bytes = capture_satellite_html("111.286131,30.693383", api_key="", zoom=18)
    if img_bytes:
        Image.open(BytesIO(img_bytes)).save("sate_map.png")
        print("saved sate_map.png")
    else:
        print("failed")
