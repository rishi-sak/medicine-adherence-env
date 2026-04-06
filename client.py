from typing import Dict, Any
import requests


class OpenEnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def reset(self) -> Dict[str, Any]:
        response = requests.post(f"{self.base_url}/reset")
        return response.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(f"{self.base_url}/step", json=action)
        return response.json()

    def state(self) -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/state")
        return response.json()