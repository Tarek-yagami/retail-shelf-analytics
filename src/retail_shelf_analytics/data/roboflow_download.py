"""Download a Roboflow dataset version and the full-resolution originals of its photos."""

import io
import os
import shutil
import time
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from retail_shelf_analytics.config import REPO_ROOT, RoboflowSource
from retail_shelf_analytics.data.roboflow_import import export_photo_key, raw_photo_key

API_URL = "https://api.roboflow.com"
API_KEY_VARIABLE = "ROBOFLOW_API_KEY"
EXPORT_FORMAT = "yolov8"
SEARCH_PAGE_SIZE = 100
TIMEOUT_SECONDS = 120
EXPORT_POLL_SECONDS = 5
EXPORT_MAX_POLLS = 60


class RoboflowError(RuntimeError):
    pass


def load_api_key() -> str:
    load_dotenv(REPO_ROOT / ".env")
    key = os.environ.get(API_KEY_VARIABLE)
    if not key:
        raise RoboflowError(f"Set {API_KEY_VARIABLE}=<your private API key> in {REPO_ROOT / '.env'}")
    return key


class RoboflowClient:
    def __init__(self, api_key: str, source: RoboflowSource) -> None:
        self._api_key = api_key
        self._project_url = f"{API_URL}/{source.workspace}/{source.project}"
        self._session = requests.Session()
        retries = Retry(total=5, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=None)
        self._session.mount("https://", HTTPAdapter(max_retries=retries))

    def _request(self, step: str, method: str, url: str, **kwargs) -> requests.Response:
        """Send a request; the key is added only for API calls and masked in every error."""
        params = {"api_key": self._api_key} if url.startswith(API_URL) else None
        try:
            response = self._session.request(method, url, params=params, timeout=TIMEOUT_SECONDS, **kwargs)
            response.raise_for_status()
        except requests.RequestException as error:
            raise RoboflowError(f"{step} failed: {str(error).replace(self._api_key, '***')}") from None
        return response

    def export_link(self, version: int) -> str:
        url = f"{self._project_url}/{version}/{EXPORT_FORMAT}"
        for _ in range(EXPORT_MAX_POLLS):
            export = self._request("Requesting the export", "GET", url).json()
            if "export" in export:
                return export["export"]["link"]
            time.sleep(EXPORT_POLL_SECONDS)  # Roboflow is still generating the export
        raise RoboflowError(f"Roboflow did not finish exporting version {version} in time; try again later.")

    def list_images(self) -> list[dict]:
        images = []
        while True:
            body = {"offset": len(images), "limit": SEARCH_PAGE_SIZE, "fields": ["id", "name"]}
            page = self._request("Listing images", "POST", f"{self._project_url}/search", json=body).json()
            images += page["results"]
            if not page["results"] or len(images) >= page["total"]:
                return images

    def original_url(self, image_id: str) -> str:
        image = self._request("Reading image details", "GET", f"{self._project_url}/images/{image_id}").json()
        return image["image"]["urls"]["original"]

    def download(self, step: str, url: str) -> bytes:
        return self._request(step, "GET", url).content


def download_version(client: RoboflowClient, version: int, export_dir: Path) -> None:
    archive = client.download("Downloading the export", client.export_link(version))
    if export_dir.exists():
        shutil.rmtree(export_dir)
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        zipped.extractall(export_dir)


def download_missing_originals(client: RoboflowClient, export_dir: Path, photos_dir: Path) -> list[Path]:
    """Save the original of every exported photo that is not yet in photos_dir, and return the new files."""
    exported = {export_photo_key(p) for p in export_dir.glob("*/images/*")}
    photos_dir.mkdir(parents=True, exist_ok=True)
    missing = exported - {raw_photo_key(p) for p in photos_dir.iterdir()}
    saved = []
    for image in client.list_images():
        if raw_photo_key(Path(image["name"])) in missing:
            path = photos_dir / image["name"]
            path.write_bytes(client.download(f"Downloading {image['name']}", client.original_url(image["id"])))
            saved.append(path)
    return saved
