from playwright.async_api._generated import BrowserContext


from playwright.async_api import async_playwright
import asyncio

# HTML模板，使用占位符{lng}、{lat}
HTML_TEMPLATE = """<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="initial-scale=2.0, user-scalable=no, width=device-width">
    <link rel="stylesheet" href="https://a.amap.com/jsapi_demos/static/demo-center/css/demo-center.css" />
    <title>地图显示</title>
    <style>
        html,
        body,
        #container {{
          width: 100%;
          height: 100%;
        }}
    </style>
</head>
<body>
<div id="container"></div>
<script src="https://webapi.amap.com/maps?v=2.0&key={api_key}"></script>
<script>
    var map = new AMap.Map('container', {{
        viewMode: '2D',
        zoom: {zoom},
        center: [{lon}, {lat}]
    }});
</script>
</body>
</html>
"""

async def capture_map_html(api_key: str,lon: float, lat: float, zoom: int = 19, width: int = 500, height: int = 500) -> bytes | None:
    """截取高德地图截图，并返回图片byte数据"""
    # 填充模板
    html_str = HTML_TEMPLATE.format(api_key=api_key, lon=lon, lat=lat, zoom=zoom)
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                executable_path="path/to/chrome-linux/chrome",
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            context: BrowserContext = await browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
                device_scale_factor=2,
                viewport={"width": width, "height": height}
            )
            page = await context.new_page()
            await page.set_content(html_str)  # 直接用HTML字符串
            await page.wait_for_timeout(3000)  # 等地图渲染
            screenshot_bytes = await page.screenshot()  # 返回字节数据
            await browser.close()
            return screenshot_bytes
    except Exception as e:
        print("???????", e)
        return None

# 示例调用
async def main():
    img_bytes = await capture_map_html(111.286131, 30.693383)
    # 假如你要保存到文件：
    with open("map.png", "wb") as f:
        f.write(img_bytes)
    print("截图已保存为 map.png")

if __name__ == "__main__":
    asyncio.run(main())
