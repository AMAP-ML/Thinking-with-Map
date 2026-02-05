from PIL.ImageFile import ImageFile
import os
from typing import Dict, Optional, Union
import uuid

import requests
from PIL import Image
from io import BytesIO
import asyncio
import nest_asyncio
nest_asyncio.apply()

from qwen_agent.llm.schema import ContentItem
from qwen_agent.tools.base import BaseTool, BaseToolWithFileAccess, register_tool
from qwen_agent.tools.amap_static_map_async import capture_map_html
from qwen_agent.tools.amap_satellite_map_async import capture_satellite_html

AMAP_API_KEY = 'xxxxx'


@register_tool('amap_input_tips')
class AmapInputTips(BaseTool):
    name = 'amap_input_tips'
    description = '搜索中国地点，获取相关POI建议'
    parameters = {
        'type': 'object',
        'properties': {
            'keywords': {
                'description': '想要搜索的任意关键词',
                'type': 'string',
            },
        },
        'required': ['keywords'],
    }

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.url = 'https://restapi.amap.com/v3/assistant/inputtips?keywords={keywords}&key={key}'

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)

        keywords = params['keywords']
        response = requests.get(self.url.format(keywords=keywords, key=AMAP_API_KEY))
        data = response.json()

        if data['status'] != '1':
            raise RuntimeError(data)

        tips = []
        for tip in data.get("tips", []):
            tips.append({
                "id": tip.get("id"),
                "name": tip.get("name"),
                "address": tip.get("address"),
                "typecode": tip.get("typecode"),
                "city": tip.get("city"),
            })
        
        return {
            "tips": tips,
        }


@register_tool('amap_keyword_search')
class AmapKeywordSearch(BaseTool):
    name = 'amap_keyword_search'
    description = '根据关键词进行搜索，搜索中国的POI'
    parameters = {
        'type': 'object',
        'properties': {
            'keywords': {
                'description': '需要查询的关键词',
                'type': 'string',
            },
        },
        'required': ['keywords'],
    }

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.url = 'https://restapi.amap.com/v5/place/text?keywords={keywords}&page_size=25&key={key}'

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)

        keywords = params['keywords']
        response = requests.get(self.url.format(keywords=keywords, key=AMAP_API_KEY))
        data = response.json()

        if data['status'] != '1':
            raise RuntimeError(data)
        
        suggestion_cities = []
        if data.get("suggestion", {}).get("cities"):
            for city in data["suggestion"]["cities"]:
                suggestion_cities.append({"name": city.get("name")})

        pois = []
        for poi in data.get("pois", []):
            pois.append({
                "id": poi.get("id"),
                "name": poi.get("name"),
                "address": poi.get("address"),
                "typecode": poi.get("typecode")
            })
        
        return {
            "suggestion": {
                "keywords": data.get("suggestion", {}).get("keywords"),
                "cities": suggestion_cities
            },
            "pois": pois
        }


@register_tool('amap_poi_detail')
class AmapPOISearch(BaseTool):
    name = 'amap_poi_detail'
    description = '根据中国的POI ID获取详细位置信息'
    parameters = {
        'type': 'object',
        'properties': {
            'id': {
                'description': '需要查询的POI ID',
                'type': 'string',
            },
        },
        'required': ['id'],
    }

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.url = 'https://restapi.amap.com/v5/place/detail?id={id}&key={key}'

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)

        id = params['id']
        response = requests.get(self.url.format(id=id, key=AMAP_API_KEY))
        data = response.json()

        if data['status'] != '1':
            raise RuntimeError(data)
        
        if not data.get("pois"):
            return {"error": "No POI found"}
            
        poi = data["pois"][0]
        result = {
            "id": poi.get("id"),
            "name": poi.get("name"),
            "location": poi.get("location"),
            "address": poi.get("address"),
            "business_area": poi.get("business_area"),
            "city": poi.get("cityname"),
            "type": poi.get("type"),
            "alias": poi.get("alias")
        }
        
        # Add biz_ext data if available
        if poi.get("biz_ext"):
            result.update(poi["biz_ext"])
            
        return result


@register_tool('amap_static_map')
class AmapStaticMap(BaseToolWithFileAccess):
    name = 'amap_static_map'
    description = '根据经纬度，返回一个静态地图，包含附近所有区域，以便近一步判断该地点。注意只能查询中国区域'
    parameters = {
        'type': 'object',
        'properties': {
            'latitude': {
                'description': '查询地点的纬度，用正/负值表示北/南纬，例如39.9042',
                'type': 'number',
            },
            'longitude': {
                'description': '查询地点的经度，用正/负值表示东/西经，例如116.4074',
                'type': 'number',
            },
        },
        'required': ['latitude', 'longitude'],
    }

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs):
        params = self._verify_json_format_args(params)
        latitude = params['latitude']
        longitude = params['longitude']

        loop = asyncio.get_event_loop()
        data_bytes = loop.run_until_complete(
            capture_map_html(api_key=AMAP_API_KEY, lon=longitude, lat=latitude)
        )

        if not data_bytes:
            raise RuntimeError("截图失败或未返回数据")

        img = Image.open(BytesIO(data_bytes))
        output_path = os.path.abspath(os.path.join(self.work_dir, f'{uuid.uuid4()}.png'))
        img.save(output_path)
        return [ContentItem(image=output_path)]


@register_tool('amap_satellite_map')
class AmapSatelliteMap(BaseToolWithFileAccess):
    name = 'amap_satellite_map'
    description = '根据经纬度，返回一个附近的卫星地图，以便近一步判断该地点。注意只能查询中国区域'
    parameters = {
        'type': 'object',
        'properties': {
            'latitude': {
                'description': '查询地点的纬度，用正/负值表示北/南纬，例如39.9042',
                'type': 'number',
            },
            'longitude': {
                'description': '查询地点的经度，用正/负值表示东/西经，例如116.4074',
                'type': 'number',
            },
        },
        'required': ['latitude', 'longitude'],
    }

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs):
        params = self._verify_json_format_args(params)
        latitude = params['latitude']
        longitude = params['longitude']

        loop = asyncio.get_event_loop()
        data_bytes: bytes | None = loop.run_until_complete(
            capture_satellite_html(api_key=AMAP_API_KEY, query=f"{longitude},{latitude}")
        )

        if not data_bytes:
            raise RuntimeError("截图失败或未返回数据")

        img = Image.open(BytesIO(data_bytes))
        output_path = os.path.abspath(os.path.join(self.work_dir, f'{uuid.uuid4()}.png'))
        img.save(output_path)
        return [ContentItem(image=output_path)]
        
