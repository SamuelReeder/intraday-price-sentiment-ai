import requests
from typing import Dict, Any


def get_data(url: str) -> Dict[str, Any]:
    r = requests.get(url)
    data = r.json()
    return data