# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any, List, Union, Optional, Dict

import requests
from PIL import Image
from io import BytesIO
import uuid

from qwen_agent.llm.schema import ContentItem
from qwen_agent.tools.base import BaseTool, BaseToolWithFileAccess, register_tool

SERPER_API_KEY = os.getenv('SERPER_API_KEY', '')
SERPER_URL = os.getenv('SERPER_URL', 'https://google.serper.dev/maps')

SERPER_API_KEY = ""

GOOGLE_MAP_API_KEY = ""

@register_tool('google_map_search', allow_overwrite=True)
class GoogleMapSearch(BaseTool):
    name = 'google_map_search'
    # description = "Search for information from the internet."
    description = "Search location (for precise latitude and longtitude) in Google Map."
    parameters = {
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
            }
        },
        'required': ['query'],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        query = params['query']

        search_results = self.search(query)
        formatted_results = self._format_results(search_results)
        return formatted_results

    @staticmethod
    def search(query: str) -> List[Any]:
        if not SERPER_API_KEY:
            raise ValueError(
                'SERPER_API_KEY is None! Please Apply for an apikey from https://serper.dev and set it as an environment variable by `export SERPER_API_KEY=xxxxxx`'
            )
        headers = {'Content-Type': 'application/json', 'X-API-KEY': SERPER_API_KEY}
        payload = {'q': query}
        response = requests.post(SERPER_URL, json=payload, headers=headers)
        response.raise_for_status()

        return response.json()['places']

    @staticmethod
    def _format_results(search_results: List[dict]) -> str:
        lines = []
        for i, doc in enumerate(search_results, 1):
            line = (
                f"[{i}] {doc.get('title', '')}\n"
                f"Address: {doc.get('address', '')}\n"
                f"latitude: {doc.get('latitude', '')}\n"
                f"longitude: {doc.get('longitude', '')}\n"
                # f"Rating: {doc.get('rating', '')} ({doc.get('ratingCount', '')} reviews)\n"
                f"Type: {', '.join(doc.get('types', []))}\n"
                # f"Phone: {doc.get('phoneNumber', '')}\n"
                # f"Website: {doc.get('website', '')}\n"
                f"Description: {doc.get('description', '')}"
            )
            lines.append(line)
        content = '```\n' + '\n\n'.join(lines) + '\n```'
        return content


@register_tool('google_static_map')
class GoogleStaticMap(BaseToolWithFileAccess):
    name = 'google_static_map'
    description = 'Based on the input latitude and longitude, return a static map that includes all nearby areas for further assessment of the location. Note: Only non-China regions can be queried.'
    parameters = {
        'type': 'object',
        'properties': {
            'latitude': {
                'description': 'Latitude of the query location, using positive/negative values to indicate north/south latitude, for example, 40.7147',
                'type': 'number',
            },
            'longitude': {
                'description': 'Longtitude of the query location, using positive/negative values to indicate east/west longitude, for example, -73.9986',
                'type': 'number',
            },
        },
        'required': ['latitude', 'longitude'],
    }

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.url = 'https://maps.googleapis.com/maps/api/staticmap?center={center}&zoom=19&size=500x500&maptype=roadmap&markers=color:red%7Clabel:S%7C{markers}&key={key}'

    def call(self, params: Union[str, dict], **kwargs):
        params = self._verify_json_format_args(params)

        latitude = params['latitude']
        longitude = params['longitude']
        response = requests.get(self.url.format(center=f"{latitude},{longitude}", markers=f"{latitude},{longitude}", key=GOOGLE_MAP_API_KEY))
        data = Image.open(BytesIO(response.content))

        if response.status_code != 200:
            raise RuntimeError(data)
        
        output_path = os.path.abspath(os.path.join(self.work_dir, f'{uuid.uuid4()}.png'))
        print("save dir", output_path)
        data.save(output_path)
            
        return [ContentItem(image=output_path)]


if __name__ == '__main__':
    print(SERPER_API_KEY)
    static_map = GoogleStaticMap()
    params = {
        "latitude": 21.2780532,
        "longitude": -157.8267731,
    }
    static_map.call(params)